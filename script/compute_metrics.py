#!/usr/bin/env python3
"""Compute PSNR / SSIM / LPIPS from concatenated comparison images.

Each input image is assumed to be a horizontal concatenation of three equal-width
panels: [GT | RESULT | DEPTH]. The script splits each image into the GT and RESULT
panels, computes the three metrics, and writes per-image results plus a summary row
to a CSV file.

Usage:
    python script/compute_metrics.py --input_dir renders/ --output metrics.csv

    # With sky masking (depth > threshold treated as sky)
    python script/compute_metrics.py --input_dir renders/ --depth_dir depth/

Dependencies:
    pip install torch torchvision scikit-image lpips Pillow
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Lazy-load heavy deps so import errors are reported clearly
# ---------------------------------------------------------------------------

def _require(pkg, pip_name=None):
    import importlib
    try:
        return importlib.import_module(pkg)
    except ImportError:
        name = pip_name or pkg
        print(f"[ERROR] Missing package '{name}'. Install with:  pip install {name}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_psnr(gt: np.ndarray, pred: np.ndarray) -> float:
    """PSNR in dB. Images are uint8 HxWxC numpy arrays."""
    mse = np.mean((gt.astype(np.float64) - pred.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0 ** 2 / mse)


def compute_ssim(gt: np.ndarray, pred: np.ndarray) -> float:
    skimage = _require("skimage", "scikit-image")
    from skimage.metrics import structural_similarity
    kwargs = dict(data_range=255)
    if gt.ndim == 3:
        kwargs["channel_axis"] = -1
    return float(structural_similarity(gt, pred, **kwargs))


def compute_lpips(gt: np.ndarray, pred: np.ndarray, lpips_model) -> float:
    """LPIPS. Expects uint8 HxWxC numpy arrays."""
    def to_tensor(arr):
        t = torch.from_numpy(arr).float() / 255.0  # [0,1]
        t = t.permute(2, 0, 1).unsqueeze(0)         # 1xCxHxW
        t = t * 2.0 - 1.0                            # [-1,1]
        return t
    with torch.no_grad():
        score = lpips_model(to_tensor(gt), to_tensor(pred))
    return float(score.item())


# ---------------------------------------------------------------------------
# Image splitting
# ---------------------------------------------------------------------------

def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 HxWx3 to grayscale HxWx3 (all channels equal)."""
    gray = (0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def split_image(img: np.ndarray):
    """Split a horizontal triptych [GT | RESULT | DEPTH] into (gt, result).

    Photographer mask: pixels where RESULT == (0,0,0) are set to black in GT
    so they contribute zero error to all metrics.
    """
    W = img.shape[1]
    assert W % 3 == 0, (
        f"Image width {W} is not divisible by 3. "
        "Expected a [GT | RESULT | DEPTH] concatenation."
    )
    w = W // 3
    gt     = img[:, :w].copy()
    result = img[:, w:2*w].copy()

    black_mask = (result == 0).all(axis=-1)  # (H, W) bool
    gt[black_mask] = 0

    return gt, result


def apply_sky_mask(gt: np.ndarray, result: np.ndarray, depth: np.ndarray,
                   sky_depth_threshold: float):
    """Return copies of gt/result with sky pixels (depth > threshold) zeroed in GT.

    Sky pixels in RESULT are already rendered as sky color; replacing the
    corresponding GT with black makes them contribute zero error.
    """
    gt_sky = gt.copy()
    sky_mask = depth > sky_depth_threshold          # (H, W) bool
    gt_sky[sky_mask] = result[sky_mask]             # GT <- RESULT so error = 0
    sky_pct = sky_mask.mean() * 100
    return gt_sky, result, sky_pct


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_images(input_dir: str):
    """Return sorted list of image paths from a directory or glob pattern."""
    if os.path.isdir(input_dir):
        paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
            paths.extend(glob.glob(os.path.join(input_dir, ext)))
        return sorted(set(paths))
    else:
        return sorted(glob.glob(input_dir))


def write_csv(path: str, rows: list, has_lpips: bool):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "psnr", "ssim", "lpips"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved to {path}")


def summarize(label: str, psnr_all, ssim_all, lpips_all):
    mean_psnr  = np.mean(psnr_all)
    mean_ssim  = np.mean(ssim_all)
    mean_lpips = np.mean(lpips_all) if lpips_all else float("nan")
    print(f"\n--- {label} (n={len(psnr_all)}) ---")
    print(f"  PSNR:  {mean_psnr:.4f} dB")
    print(f"  SSIM:  {mean_ssim:.4f}")
    if lpips_all:
        print(f"  LPIPS: {mean_lpips:.4f}")
    return mean_psnr, mean_ssim, mean_lpips


def main():
    parser = argparse.ArgumentParser(
        description="Compute PSNR/SSIM/LPIPS from [GT|RESULT|DEPTH] concatenated images"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Directory containing images, or a glob pattern (e.g. 'renders/*.png')",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV file path (default: <input_dir>/../metrics.csv)",
    )
    parser.add_argument(
        "--lpips_net", type=str, default="alex", choices=["alex", "vgg"],
        help="LPIPS backbone network (default: alex)",
    )
    parser.add_argument(
        "--no_lpips", action="store_true",
        help="Skip LPIPS computation (faster, no lpips package needed)",
    )
    parser.add_argument(
        "--step", type=int, default=1,
        help="Compute metrics every N images (default: 1, i.e. all images)",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Show a window with masked GT and RESULT for each computed image",
    )
    parser.add_argument(
        "--depth_dir", type=str, default=None,
        help="Directory containing depth .npy files (same stem as images + '_depth.npy'). "
             "When provided, also outputs metrics_wo_sky.csv with sky pixels excluded.",
    )
    parser.add_argument(
        "--sky_depth", type=float, default=2.0,
        help="Depth threshold above which a pixel is treated as sky (default: 2.0)",
    )
    parser.add_argument(
        "--grayscale", action="store_true",
        help="Convert images to grayscale before computing metrics",
    )
    args = parser.parse_args()

    # Resolve default output path
    if args.output is None:
        input_abs = os.path.abspath(args.input_dir.rstrip("/\\*?"))
        if not os.path.isdir(input_abs):
            input_abs = os.path.dirname(input_abs)
        args.output = os.path.join(os.path.dirname(input_abs), "metrics.csv")

    output_wo_sky = os.path.join(os.path.dirname(args.output), "metrics_wo_sky.csv")

    image_paths = collect_images(args.input_dir)
    if not image_paths:
        print(f"[ERROR] No images found in: {args.input_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(image_paths)} images.")

    # Load LPIPS model once
    lpips_model = None
    if not args.no_lpips:
        lpips_lib = _require("lpips")
        lpips_model = lpips_lib.LPIPS(net=args.lpips_net)
        lpips_model.eval()

    if args.visualize:
        import cv2

    rows      = []
    rows_sky  = []
    psnr_all,     ssim_all,     lpips_all     = [], [], []
    psnr_sky_all, ssim_sky_all, lpips_sky_all = [], [], []
    brightness_ratios = []  # TEMP: gt/result brightness ratio per frame

    sampled_paths = image_paths[::args.step]
    print(f"Processing {len(sampled_paths)} / {len(image_paths)} images (step={args.step}).")
    if args.depth_dir:
        print(f"Sky masking enabled: depth > {args.sky_depth} -> sky")

    for path in sampled_paths:
        img = np.array(Image.open(path).convert("RGB"))
        try:
            gt, result = split_image(img)
        except AssertionError as e:
            print(f"[SKIP] {os.path.basename(path)}: {e}")
            continue

        # result = np.clip(result.astype(np.float32) * 1.021, 0, 255).astype(np.uint8)

        if args.grayscale:
            gt     = to_grayscale(gt)
            result = to_grayscale(result)

        # TEMP: compute gt/result brightness ratio over valid pixels
        valid_mask = (result > 0).any(axis=-1)  # (H, W)
        if valid_mask.any():
            gt_lum     = gt[valid_mask].astype(np.float64).mean(axis=-1)
            result_lum = result[valid_mask].astype(np.float64).mean(axis=-1)
            nonzero    = result_lum > 0
            if nonzero.any():
                ratio = (gt_lum[nonzero] / result_lum[nonzero]).mean()
                brightness_ratios.append(ratio)

        fname = os.path.basename(path)

        # --- sky-masked version ---
        gt_sky = None
        sky_pct = 0.0
        if args.depth_dir:
            stem = os.path.splitext(fname)[0]
            depth_path = os.path.join(args.depth_dir, stem + "_depth.npy")
            if not os.path.exists(depth_path):
                print(f"  [WARN] depth not found: {depth_path}, skipping sky mask for this image")
            else:
                depth = np.load(depth_path)
                gt_sky, _, sky_pct = apply_sky_mask(gt, result, depth, args.sky_depth)

        # --- visualization ---
        if args.visualize:
            gt_vis = gt_sky if gt_sky is not None else gt
            vis = np.concatenate([gt_vis, result], axis=1)
            vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            cv2.imshow("GT (masked)  |  RESULT", vis_bgr)
            cv2.waitKey(1)

        # --- base metrics (skipped when depth_dir is provided) ---
        log = f"  {fname:<40}"
        if not args.depth_dir:
            psnr        = compute_psnr(gt, result)
            ssim        = compute_ssim(gt, result)
            lpips_score = compute_lpips(gt, result, lpips_model) if lpips_model else float("nan")

            psnr_all.append(psnr)
            ssim_all.append(ssim)
            if lpips_model:
                lpips_all.append(lpips_score)

            rows.append({
                "filename": fname,
                "psnr":     f"{psnr:.4f}",
                "ssim":     f"{ssim:.4f}",
                "lpips":    f"{lpips_score:.4f}" if lpips_model else "N/A",
            })

            log += (f"  PSNR={psnr:7.3f} dB  SSIM={ssim:.4f}"
                    + (f"  LPIPS={lpips_score:.4f}" if lpips_model else ""))

        # --- wo-sky metrics ---
        if gt_sky is not None:
            psnr_s        = compute_psnr(gt_sky, result)
            ssim_s        = compute_ssim(gt_sky, result)
            lpips_s       = compute_lpips(gt_sky, result, lpips_model) if lpips_model else float("nan")

            psnr_sky_all.append(psnr_s)
            ssim_sky_all.append(ssim_s)
            if lpips_model:
                lpips_sky_all.append(lpips_s)

            rows_sky.append({
                "filename": fname,
                "psnr":     f"{psnr_s:.4f}",
                "ssim":     f"{ssim_s:.4f}",
                "lpips":    f"{lpips_s:.4f}" if lpips_model else "N/A",
            })
            log += (f"  | wo_sky({sky_pct:.1f}%)  PSNR={psnr_s:7.3f} dB  SSIM={ssim_s:.4f}"
                    + (f"  LPIPS={lpips_s:.4f}" if lpips_model else ""))

        print(log)

    if args.visualize:
        cv2.destroyAllWindows()

    # TEMP: print brightness ratio summary
    if brightness_ratios:
        print(f"\n[TEMP] GT/Result brightness ratio over valid pixels: "
              f"mean={np.mean(brightness_ratios):.4f}  "
              f"std={np.std(brightness_ratios):.4f}  "
              f"n={len(brightness_ratios)}")

    if not rows and not rows_sky:
        print("[ERROR] No valid images were processed.", file=sys.stderr)
        sys.exit(1)

    # Summaries
    if rows:
        mean_psnr, mean_ssim, mean_lpips = summarize("All pixels", psnr_all, ssim_all, lpips_all)
        rows.append({
            "filename": "MEAN",
            "psnr":     f"{mean_psnr:.4f}",
            "ssim":     f"{mean_ssim:.4f}",
            "lpips":    f"{mean_lpips:.4f}" if lpips_all else "N/A",
        })
        write_csv(args.output, rows, bool(lpips_model))

    if rows_sky:
        mean_psnr_s, mean_ssim_s, mean_lpips_s = summarize(
            f"Without sky (threshold={args.sky_depth})", psnr_sky_all, ssim_sky_all, lpips_sky_all)
        rows_sky.append({
            "filename": "MEAN",
            "psnr":     f"{mean_psnr_s:.4f}",
            "ssim":     f"{mean_ssim_s:.4f}",
            "lpips":    f"{mean_lpips_s:.4f}" if lpips_sky_all else "N/A",
        })
        write_csv(output_wo_sky, rows_sky, bool(lpips_model))


if __name__ == "__main__":
    main()
