"""
Generate binary sky masks from rendered point cloud images.

Input images (from project_pointcloud.py):
  - Alpha == 0          : sky (upward-looking rays, no points)
  - Alpha == 255        : foreground / ground (point cloud or ground_white)

Pipeline per image:
  1. Binarize: alpha > 0 → foreground (255), alpha == 0 → sky (0)
  2. Dilate foreground by morph_n pixels (merges nearby spheres)
  3. Compute Z = pixel area of one minimum sphere (5-px cross) after dilation
  4. CC filter: remove components with area < n_factor * Z
     (n_factor=2 means need at least 2 merged spheres → isolated spheres removed)
  5. Invert: sky=255, foreground=0
     (foreground boundary stays expanded from step 2 — conservative for sky mask)
  6. [optional] Color refinement: among alpha==0 pixels, additionally mark as sky
     those whose original RGB has large B channel and high overall brightness.
     Final mask = step-5 mask | color sky pixels.

Usage:
    python make_sky_mask.py \
        --input_dir  /path/to/projected_images \
        --output_dir /path/to/sky_masks \
        [--morph_n 5]          # dilation radius in pixels (default: 5)
        [--n_factor 2]         # keep CC with area >= n_factor * Z (default: 2)
        [--kernel_shape ellipse]
        [--save_intermediate]  # also save after dilation, before CC filter
        [--orig_dir /path/to/original/images]  # overlay visualization + color sky detection
        [--sky_color]          # enable color-based sky refinement (requires --orig_dir)
        [--sky_b_thresh 100]   # B channel threshold for sky color detection (default: 100)
        [--sky_gray_thresh 100] # grayscale threshold for sky color detection (default: 100)
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
    parser.add_argument("--morph_n", type=int, default=9,
                        help="Dilation radius in pixels (default: 9)")
    parser.add_argument("--n_factor", type=float, default=2.0,
                        help="Keep CC with area >= n_factor * Z, where Z is one dilated min-sphere area (default: 2)")
    parser.add_argument("--kernel_shape", choices=["ellipse", "rect"], default="ellipse")
    parser.add_argument("--save_intermediate", action="store_true",
                        help="Also save mask after dilation but before CC filter, "
                             "in <output_dir>_intermediate/")
    parser.add_argument("--orig_dir", default=None,
                        help="Original camera image directory. If given, save overlay "
                             "visualizations in <output_dir>_overlay/")
    parser.add_argument("--sky_color", action="store_true",
                        help="Enable color-based sky refinement (requires --orig_dir)")
    parser.add_argument("--sky_b_thresh", type=int, default=220,
                        help="B channel threshold for sky color detection (default: 220)")
    parser.add_argument("--sky_gray_thresh", type=int, default=200,
                        help="Grayscale threshold for sky color detection (default: 200)")
    return parser.parse_args()


def make_kernel(n, shape):
    size = 2 * n + 1
    if shape == "ellipse":
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    else:
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))


def compute_Z(morph_n, kernel):
    """Pixel area of a minimum 5-pixel cross sphere after dilation by morph_n."""
    pad = morph_n + 2
    size = 2 * pad + 1
    cross = np.zeros((size, size), dtype=np.uint8)
    cx = cy = pad
    cross[cy, cx] = 255
    cross[cy - 1, cx] = 255
    cross[cy + 1, cx] = 255
    cross[cy, cx - 1] = 255
    cross[cy, cx + 1] = 255
    dilated = cv2.dilate(cross, kernel)
    return int((dilated > 0).sum())


def color_sky_mask(orig_bgr, alpha_zero, b_thresh, gray_thresh):
    """Among alpha==0 pixels, detect sky by color: large B channel and high brightness.

    Args:
        orig_bgr:    original image in BGR (H, W, 3) uint8
        alpha_zero:  bool mask (H, W), True where alpha==0 in projected image
        b_thresh:    minimum B channel value to consider sky
        gray_thresh: minimum grayscale value to consider sky
    Returns:
        bool mask (H, W), True = sky by color
    """
    b = orig_bgr[:, :, 0].astype(np.float32)       # B channel (BGR)
    gray = orig_bgr.mean(axis=2)                     # overall brightness
    looks_like_sky = (b > b_thresh) & (gray > gray_thresh)
    return alpha_zero & looks_like_sky


def process_image(img, morph_n, n_factor, kernel_shape, return_intermediate=False,
                  orig_bgr=None, sky_color=False, sky_b_thresh=100, sky_gray_thresh=100):
    # 1. Binarize: alpha > 0 → 255 (foreground), alpha == 0 → 0 (sky)
    #    Fall back to max(RGB) > 0 for legacy RGB images without alpha channel.
    if img.ndim == 3 and img.shape[2] == 4:
        alpha_zero = img[:, :, 3] == 0  # save for color refinement
        fg = (~alpha_zero).astype(np.uint8) * 255
    else:
        alpha_zero = img.max(axis=2) == 0
        fg = (~alpha_zero).astype(np.uint8) * 255

    k_n = make_kernel(morph_n, kernel_shape)

    # 2. Dilate foreground — merges nearby spheres into larger components
    fg = cv2.dilate(fg, k_n)

    # Intermediate result (after dilation, before CC filter)
    intermediate = 255 - fg if return_intermediate else None

    # 3. Compute Z: area of one minimum sphere after dilation
    Z = compute_Z(morph_n, k_n)
    threshold = n_factor * Z

    # 4. CC filter: remove components smaller than n_factor * Z
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    filtered = np.zeros_like(fg)
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= threshold:
            filtered[labels == lbl] = 255
    fg = filtered

    # 5. Invert: sky=255, foreground=0
    sky_mask = 255 - fg

    # 6. Color refinement: union with color-based sky among alpha==0 pixels
    if sky_color and orig_bgr is not None:
        if orig_bgr.shape[:2] != img.shape[:2]:
            orig_bgr = cv2.resize(orig_bgr, (img.shape[1], img.shape[0]))
        color_sky = color_sky_mask(orig_bgr, alpha_zero, sky_b_thresh, sky_gray_thresh)
        sky_mask = np.where(color_sky, np.uint8(255), sky_mask)

    return (sky_mask, intermediate) if return_intermediate else sky_mask


ORIG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]


def find_orig_image(orig_dir, rel_path):
    """Find original image with same relative path but possibly different extension."""
    base = os.path.splitext(rel_path)[0]
    for ext in ORIG_EXTENSIONS:
        candidate = os.path.join(orig_dir, base + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def make_overlay(orig_bgr, sky_mask, alpha=0.5):
    """Tint sky pixels (mask==255) with red on the original image."""
    overlay = orig_bgr.copy()
    sky = sky_mask == 255
    overlay[sky] = (orig_bgr[sky] * (1 - alpha) +
                    np.array([0, 0, 200], dtype=np.float32) * alpha).astype(np.uint8)
    return overlay


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

    # Report Z for reference
    k_n = make_kernel(args.morph_n, args.kernel_shape)
    Z = compute_Z(args.morph_n, k_n)
    print(f"morph_n={args.morph_n}, kernel={args.kernel_shape}, "
          f"Z={Z}px, threshold={args.n_factor}*Z={args.n_factor * Z:.0f}px")

    inter_dir = os.path.join(os.path.dirname(args.output_dir),
                             os.path.basename(args.output_dir) + "_intermediate") \
                if args.save_intermediate else None
    if inter_dir:
        os.makedirs(inter_dir, exist_ok=True)
        print(f"Intermediate results → {inter_dir}")

    overlay_dir = os.path.join(os.path.dirname(args.output_dir),
                               os.path.basename(args.output_dir) + "_overlay") \
                  if args.orig_dir else None
    if overlay_dir:
        os.makedirs(overlay_dir, exist_ok=True)
        print(f"Overlay results → {overlay_dir}")

    for img_path in tqdm(image_paths, desc="Processing"):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  WARNING: cannot read {img_path}, skipping")
            continue

        rel = os.path.relpath(img_path, args.input_dir)

        orig_bgr = None
        if args.orig_dir is not None:
            orig_path = find_orig_image(args.orig_dir, rel)
            if orig_path is not None:
                orig_bgr = cv2.imread(orig_path)

        result = process_image(img, args.morph_n, args.n_factor, args.kernel_shape,
                               return_intermediate=args.save_intermediate,
                               orig_bgr=orig_bgr, sky_color=args.sky_color,
                               sky_b_thresh=args.sky_b_thresh,
                               sky_gray_thresh=args.sky_gray_thresh)
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

        if overlay_dir:
            if orig_bgr is None:
                print(f"  WARNING: no original image found for {rel}, skipping overlay")
            else:
                orig = orig_bgr
                if orig.shape[:2] != sky_mask.shape[:2]:
                    orig = cv2.resize(orig, (sky_mask.shape[1], sky_mask.shape[0]))
                ov = make_overlay(orig, sky_mask)
                ov_path = os.path.join(overlay_dir, rel)
                os.makedirs(os.path.dirname(ov_path), exist_ok=True)
                cv2.imwrite(ov_path, ov)

    print(f"\nDone. Masks saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
