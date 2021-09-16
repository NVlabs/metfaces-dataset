# Copyright 2020 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import numpy as np
import os
import PIL.Image
import scipy.ndimage
from tqdm import tqdm

_examples = '''examples:

  # Run x, y, z
  python %(prog)s --output=tmp
'''

def extract_face(face, source_images, output_dir, rng, target_size=1024, supersampling=4, enable_padding=True, random_shift=0.0, retry_crops=False, rotate_level=True):
    def rot90(v) -> np.ndarray:
        return np.array([-v[1], v[0]])

    # Sanitize facial landmarks.
    face_spec = face['face_spec']
    landmarks = (np.float32(face_spec['landmarks']) + 0.5) * face_spec['shrink']
    assert landmarks.shape == (68, 2)
    lm_eye_left      = landmarks[36 : 42]  # left-clockwise
    lm_eye_right     = landmarks[42 : 48]  # left-clockwise
    lm_mouth_outer   = landmarks[48 : 60]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_left   = lm_mouth_outer[0]
    mouth_right  = lm_mouth_outer[6]
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    if rotate_level:
        # Orient according to tilt of the input image
        x = eye_to_eye - rot90(eye_to_mouth)
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = rot90(x)
        c0 = eye_avg + eye_to_mouth * 0.1
    else:
        # Do not match the tilt in the source data, i.e., use an axis-aligned rectangle
        x = np.array([1, 0], dtype=np.float64)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c0 = eye_avg + eye_to_mouth * 0.1

    # Load.
    img = PIL.Image.open(os.path.join(source_images, face['source_path'])).convert('RGB')

    # Calculate auxiliary data.
    qsize = np.hypot(*x) * 2
    quad = np.stack([c0 - x - y, c0 - x + y, c0 + x + y, c0 + x - y])

    # Keep drawing new random crop offsets until we find one that is contained in the image
    # and does not require padding
    if random_shift != 0:
        for _ in range(1000):
            # Offset the crop rectange center by a random shift proportional to image dimension
            # and the requested standard deviation (by default 0)
            c = (c0 + np.hypot(*x)*2 * random_shift * rng.normal(0, 1, c0.shape))
            quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
            crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
            if not retry_crops or not (crop[0] < 0 or crop[1] < 0 or crop[2] >= img.width or crop[3] >= img.height):
                # We're happy with this crop (either it fits within the image, or retries are disabled)
                break
        else:
            # rejected N times, give up and move to next image
            # (does not happen in practice with the MetFaces data)
            print('rejected image %s' % face['source_path'])
            return

    # Shrink.
    shrink = int(np.floor(qsize / target_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    super_size = target_size * supersampling
    img = img.transform((super_size, super_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if target_size < super_size:
        img = img.resize((target_size, target_size), PIL.Image.ANTIALIAS)

    # Save face image.
    img.save(os.path.join(output_dir, f"{face['obj_id']}-{face['face_idx']:02d}.png"))


def main():
    parser = argparse.ArgumentParser(
        description='MetFaces dataset processing tool',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--json', help='MetFaces metadata json file path', required=True)
    parser.add_argument('--source-images', help='Location of MetFaces raw image data', required=True)
    parser.add_argument('--output-dir', help='Where to save output files')
    parser.add_argument('--random-shift', help='Standard deviation of random crop rectangle jitter', type=float, default=0.0, metavar='SHIFT')
    parser.add_argument('--retry-crops', help='Retry random shift if crop rectangle falls outside image (up to 1000 times)', dest='retry_crops', default=False, action='store_true')
    parser.add_argument('--no-rotation', help='Keep the original orientation of images', dest='no_rotation', default=False, action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    rng = np.random.RandomState(12345)  # fix the random seed for reproducibility

    with open(args.json, encoding="utf8") as fin:
        faces = json.load(fin)
        for f in tqdm(faces):
            extract_face(f, source_images=args.source_images, output_dir=args.output_dir, rng=rng,
                random_shift=args.random_shift, retry_crops=args.retry_crops, rotate_level=not args.no_rotation)

if __name__ == "__main__":
    main()
