"""
Generate binary sky masks from rendered point cloud images.

Input images (from project_pointcloud.py):
  - Black   (0,0,0)     : sky (upward-looking rays, no points)
  - Colored or white    : foreground / ground

Pipeline per image:
  1. Binarize: non-black → foreground (255), black → sky (0)
  2. Connected-component filter: remove small foreground blobs (noise in sky)
  3. Morphological: erode N → dilate 2N → erode N
     = opening (remove remaining sky noise) + closing (fill foreground holes)
  4. Invert: sky=255, non-sky=0

Usage:
    python make_sky_mask.py \
        --input_dir  /path/to/projected_images \
        --output_dir /path/to/sky_masks \
        [--morph_n 10]               # N pixels for erosion/dilation (default: 10)
        [--min_area 500]             # min foreground component area to keep (default: 500)
        [--kernel_shape ellipse]     # ellipse or rect (default: ellipse)
"""

import argparse
import os
import numpy as np
import cv2
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--morph_n", type=int, default=10,
                        help="N pixels: erode N → dilate 2N → erode N (default: 10)")
    parser.add_argument("--min_area", type=int, default=500,
                        help="Min foreground connected-component area to keep (default: 500)")
    parser.add_argument("--kernel_shape", choices=["ellipse", "rect"], default="ellipse")
    parser.add_argument("--save_intermediate", action="store_true",
                        help="Also save mask after CC filter but before morphological ops, in output_dir/../<output_dir_name>_intermediate/")
    return parser.parse_args()


def make_kernel(n, shape):
    size = 2 * n + 1
    if shape == "ellipse":
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    else:
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))


def process_image(img_bgr, morph_n, min_area, kernel_shape, return_intermediate=False):
    # 1. Binarize: non-black → 255 (foreground), pure black → 0 (sky)
    fg = (img_bgr.max(axis=2) > 0).astype(np.uint8) * 255

    # 2. Connected-component filter: remove small foreground blobs (noise in sky)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    filtered = np.zeros_like(fg)
    for lbl in range(1, num_labels):  # skip background label 0
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == lbl] = 255
    fg = filtered

    # Intermediate result: sky=255, foreground=0 (before morphological ops)
    intermediate = 255 - fg if return_intermediate else None

    # 3. Morphological: erode N → dilate 2N → erode N
    #    = opening (kills remaining sky noise) + closing (fills foreground holes)
    k_n  = make_kernel(morph_n, kernel_shape)
    k_2n = make_kernel(morph_n * 2, kernel_shape)
    fg = cv2.erode(fg, k_n)
    fg = cv2.dilate(fg, k_2n)
    fg = cv2.erode(fg, k_n)

    # 4. Invert: sky=255, foreground=0
    sky_mask = 255 - fg
    return (sky_mask, intermediate) if return_intermediate else sky_mask


def collect_images(input_dir):
    paths = []
    for root, _, files in os.walk(input_dir):
        for f in sorted(files):
            if f.lower().endswith(".png") or f.lower().endswith(".jpg"):
                paths.append(os.path.join(root, f))
    return sorted(paths)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    image_paths = collect_images(args.input_dir)
    print(f"Found {len(image_paths)} images in {args.input_dir}")
    print(f"morph_n={args.morph_n}, min_area={args.min_area}, kernel={args.kernel_shape}")

    inter_dir = os.path.join(os.path.dirname(args.output_dir),
                             os.path.basename(args.output_dir) + "_intermediate") \
                if args.save_intermediate else None
    if inter_dir:
        os.makedirs(inter_dir, exist_ok=True)
        print(f"Intermediate results → {inter_dir}")

    for img_path in tqdm(image_paths, desc="Processing"):
        img = cv2.imread(img_path)
        if img is None:
            print(f"  WARNING: cannot read {img_path}, skipping")
            continue

        rel = os.path.relpath(img_path, args.input_dir)

        result = process_image(img, args.morph_n, args.min_area, args.kernel_shape,
                               return_intermediate=args.save_intermediate)
        if args.save_intermediate:
            sky_mask, intermediate = result
            inter_path = os.path.join(inter_dir, rel)
            os.makedirs(os.path.dirname(inter_path), exist_ok=True)
            cv2.imwrite(inter_path, intermediate)
        else:
            sky_mask = result

        out_path = os.path.join(args.output_dir, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, sky_mask)

    print(f"\nDone. Masks saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
