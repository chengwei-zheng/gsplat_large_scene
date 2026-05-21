"""Stitch renders from two directories side by side.

dir1 images: 2N x N (width x height)
dir2 images: 3N x N — crop middle N x N
Result: dir1 (2N x N) | dir2-crop (N x N) = 3N x N

Matching: by the last number in the filename (e.g., _0000.png).
"""

import os
import re

import numpy as np
from PIL import Image
from tqdm import tqdm

dir1 = "/home/yiinqiang/SMBC/gsplat/results_RawCMR/Forest_hinoki_260413_L2PRO_RawCMR_opti_global03_grow02_SReg_BG114_3GPU/renders"
dir2 = "/home/yiinqiang/SMBC/gsplat/results_RawCMR/Forest_hinoki_260413_L2PRO_RawCMR_LCC/renders"
output_dir = "/home/yiinqiang/SMBC/gsplat/results_RawCMR/Forest_hinoki_260413_L2PRO_RawCMR_opti_global03_grow02_SReg_BG114_3GPU/renders_stitched"

os.makedirs(output_dir, exist_ok=True)


def last_number(filename):
    """Return the last digit sequence in a filename (without extension)."""
    stem = os.path.splitext(filename)[0]
    nums = re.findall(r'\d+', stem)
    return nums[-1] if nums else None


# Index dir2 by last number
dir2_index = {}
for fname in os.listdir(dir2):
    if not fname.lower().endswith('.png'):
        continue
    key = last_number(fname)
    if key is not None:
        dir2_index[key] = os.path.join(dir2, fname)

matched = skipped = 0
for fname in tqdm(sorted(os.listdir(dir1))):
    if not fname.lower().endswith('.png'):
        continue
    key = last_number(fname)
    if key is None or key not in dir2_index:
        skipped += 1
        continue

    img1 = np.array(Image.open(os.path.join(dir1, fname)).convert("RGB"))  # (H, W1, 3)
    img2 = np.array(Image.open(dir2_index[key]).convert("RGB"))             # (H, W2, 3)

    W1 = img1.shape[1]
    img1 = img1[:, :W1 * 2 // 3]  # left 2/3 of img1

    W2 = img2.shape[1]   # 3N
    N = W2 // 3
    crop = img2[:, N:2 * N]  # middle N x N

    result = np.concatenate([img1, crop], axis=1)
    Image.fromarray(result).save(os.path.join(output_dir, fname))
    matched += 1

print(f"Done: {matched} stitched, {skipped} skipped (no match). Output: {output_dir}")
