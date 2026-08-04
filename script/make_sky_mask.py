"""
Generate 3-class (sky / ground / other) masks from rendered point cloud images.

Input images (from project_pointcloud.py, saved as lossless PNG):
  - Alpha == 0          : sky (upward-looking rays, no points)
  - Alpha == 127        : ground (ground_white placeholder pixels); treated the
                          same as sky throughout the pipeline
  - Alpha == 255        : foreground (real projected points)

Alternative input (--from_raw_mask), e.g. a segmentation label map like
/home/yiinqiang/SMBC/data/Wakasugi_FJD_jb1_2026-06-09-16-06-58/mask_ori:
  - Grayscale value ~= raw_sky_value (default 127) : sky
  - Any other grayscale value                      : foreground
  (this mode has no ground concept; output is only sky/other)

Output mask values:
  - 255 : sky
  - 127 : ground
  - 0   : other (real structure / foreground)

Pipeline per image:
  1. Binarize: alpha == 0 or alpha == 127 → sky/ground (0), else → foreground (255)
  2. Dilate foreground by morph_n pixels (merges nearby spheres)
  2b. Morphologically close the alpha==127 "is_ground" mask by morph_n. This
     bridges small holes caused by isolated real ground points in the point
     cloud: such a point renders as ordinary foreground (alpha==255), not the
     ground_white placeholder, so without closing it is indistinguishable from
     a tree/building point and defaults to sky if later removed by the CC
     filter in step 4.
  3. Compute Z = pixel area of one minimum sphere (5-px cross) after dilation
  4. CC filter: remove components with area < n_factor * Z
     (n_factor=2 means need at least 2 merged spheres → isolated spheres removed)
  5. Assemble 3-class mask: surviving foreground → other (0); everything else →
     ground (127) if originally ground or a closed-in ground hole (step 2b),
     else sky (255)
     (foreground boundary stays expanded from step 2 — conservative for sky/ground mask)
  6. [optional] Color refinement: among alpha==0 pixels, additionally mark as sky
     those whose original RGB has large B channel and high overall brightness.
     This overrides step-5 classification (including "other") for those pixels.

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
        [--from_raw_mask]      # input_dir holds raw label masks instead of alpha-channel
                                # renders; sky = grayscale value within raw_sky_tol of
                                # raw_sky_value, instead of alpha == 0
        [--raw_sky_value 127]  # grayscale value that marks sky in raw mask mode
        [--raw_sky_tol 50]     # +/- tolerance around raw_sky_value
        [--ground_alpha_value 127] # exact alpha value treated as ground (default: 127)
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
    parser.add_argument("--from_raw_mask", action="store_true",
                        help="Treat input_dir images as raw label masks (grayscale) instead "
                             "of alpha-channel renders: sky is the grayscale value close to "
                             "--raw_sky_value, rather than alpha == 0")
    parser.add_argument("--raw_sky_value", type=int, default=127,
                        help="Grayscale value that marks sky in --from_raw_mask mode (default: 127)")
    parser.add_argument("--raw_sky_tol", type=int, default=50,
                        help="+/- tolerance around --raw_sky_value (default: 50)")
    parser.add_argument("--ground_alpha_value", type=int, default=127,
                        help="Exact alpha value treated as ground (default: 127). Input PNGs "
                             "are lossless so this can be an exact match, no tolerance needed.")
    return parser.parse_args()


SKY_VALUE = 255
GROUND_VALUE = 127
OTHER_VALUE = 0


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


def raw_mask_sky(img, sky_value, tol):
    """True where the (grayscale) raw label mask is within tol of sky_value."""
    gray = img.astype(np.float32) if img.ndim == 2 else img.astype(np.float32).mean(axis=2)
    return np.abs(gray - sky_value) <= tol


def process_image(img, morph_n, n_factor, kernel_shape, return_intermediate=False,
                  orig_bgr=None, sky_color=False, sky_b_thresh=100, sky_gray_thresh=100,
                  from_raw_mask=False, raw_sky_value=127, raw_sky_tol=10,
                  ground_alpha_value=127):
    # 1. Binarize: foreground → 255, sky/ground → 0.
    if from_raw_mask:
        # Raw label mask: sky = grayscale value close to raw_sky_value. No ground concept.
        alpha_zero = raw_mask_sky(img, raw_sky_value, raw_sky_tol)
        is_ground = np.zeros_like(alpha_zero, dtype=bool)
        fg = (~alpha_zero).astype(np.uint8) * 255
    elif img.ndim == 3 and img.shape[2] == 4:
        # Alpha == 0 → sky. Alpha == ground_alpha_value → ground (ground_white
        # placeholder), treated the same as sky. Alpha == 255 → foreground.
        alpha = img[:, :, 3]
        alpha_zero = alpha == 0  # save for color refinement
        is_ground = alpha == ground_alpha_value
        background = alpha_zero | is_ground
        fg = (~background).astype(np.uint8) * 255
    else:
        # Legacy RGB images without alpha channel: black pixels → sky. No ground concept.
        alpha_zero = img.max(axis=2) == 0
        is_ground = np.zeros_like(alpha_zero, dtype=bool)
        fg = (~alpha_zero).astype(np.uint8) * 255

    k_n = make_kernel(morph_n, kernel_shape)

    # 2. Dilate foreground — merges nearby spheres into larger components
    fg = cv2.dilate(fg, k_n)

    # Close small holes in is_ground caused by isolated real ground points: a
    # sparse ground point in the point cloud renders as ordinary foreground
    # (alpha==255, not the ground_white placeholder), so it is indistinguishable
    # from a tree/building point at this stage. If the CC filter below removes it
    # as noise, it must fall back to ground (not sky) — closing bridges these
    # small holes using their surrounding placeholder-ground pixels.
    is_ground_u8 = is_ground.astype(np.uint8) * 255
    is_ground_closed = cv2.morphologyEx(is_ground_u8, cv2.MORPH_CLOSE, k_n) > 0

    # Intermediate result (after dilation, before CC filter): 3-class, same
    # sky/ground/other scheme as the final mask, so misclassifications are
    # visible before the CC filter is even applied.
    intermediate = np.where(fg == 255, np.uint8(OTHER_VALUE),
                            np.where(is_ground_closed, np.uint8(GROUND_VALUE), np.uint8(SKY_VALUE))) \
                   if return_intermediate else None

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

    # 5. Assemble 3-class mask: surviving foreground -> other (0); everything else
    # -> ground (127) if originally ground (or closed-in ground hole), else sky (255).
    sky_mask = np.where(fg == 255, np.uint8(OTHER_VALUE),
                        np.where(is_ground_closed, np.uint8(GROUND_VALUE), np.uint8(SKY_VALUE)))

    # 6. Color refinement: force sky among alpha==0 pixels detected as sky by color
    if sky_color and orig_bgr is not None:
        if orig_bgr.shape[:2] != img.shape[:2]:
            orig_bgr = cv2.resize(orig_bgr, (img.shape[1], img.shape[0]))
        color_sky = color_sky_mask(orig_bgr, alpha_zero, sky_b_thresh, sky_gray_thresh)
        sky_mask = np.where(color_sky, np.uint8(SKY_VALUE), sky_mask)

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
    """Tint sky pixels (mask==SKY_VALUE) red and ground pixels (mask==GROUND_VALUE) green."""
    overlay = orig_bgr.copy()
    sky = sky_mask == SKY_VALUE
    ground = sky_mask == GROUND_VALUE
    overlay[sky] = (orig_bgr[sky] * (1 - alpha) +
                    np.array([0, 0, 200], dtype=np.float32) * alpha).astype(np.uint8)
    overlay[ground] = (orig_bgr[ground] * (1 - alpha) +
                       np.array([0, 200, 0], dtype=np.float32) * alpha).astype(np.uint8)
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
                               sky_gray_thresh=args.sky_gray_thresh,
                               from_raw_mask=args.from_raw_mask,
                               raw_sky_value=args.raw_sky_value,
                               raw_sky_tol=args.raw_sky_tol,
                               ground_alpha_value=args.ground_alpha_value)
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
