# batch_ca.py
# Usage: python batch_ca.py --input slices/ --output slices_fixed/ --params ca_params.json
#        python batch_ca.py --input slices/ --output slices_fixed/ --export_mask --mask_dir slices_mask/
import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm

def process(img, p, return_mask=False):
    b, g, r = cv2.split(img.astype(np.float32))
    h, w = g.shape
    cx, cy = w / 2.0, h / 2.0

    xs = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    ys = np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, w))

    def remap(ch, scale):
        mx = (xs - cx) / scale + cx
        my = (ys - cy) / scale + cy
        return cv2.remap(ch, mx, my, cv2.INTER_LINEAR)

    r_fixed = remap(r, p["r_scale"] / 1000.0)
    b_fixed = remap(b, p["b_scale"] / 1000.0)
    img_ca = cv2.merge([b_fixed, g, r_fixed]).astype(np.uint8)

    img_hsv = cv2.cvtColor(img_ca, cv2.COLOR_BGR2HSV).astype(np.float32)
    hh, s, v = cv2.split(img_hsv)

    m = max(p.get("margin", 30), 1)
    hue_mask = np.clip((hh - p["hue_lo"]) / m, 0.0, 1.0) * \
               np.clip((p["hue_hi"] - hh) / m, 0.0, 1.0)
    s_mask   = np.clip((s - p["sat_thresh"]) / m, 0.0, 1.0)
    v_mask   = np.clip((p.get("val_thresh", 150) - v) / m, 0.0, 1.0)

    mask = hue_mask * s_mask * v_mask
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    alpha = mask * (p["strength"] / 100.0)
    s -= s * alpha
    s  = np.clip(s, 0, 255)
    v -= v * mask * (p.get("darken", 30) / 100.0)
    v  = np.clip(v, 0, 255)
    result = cv2.cvtColor(cv2.merge([hh, s, v]).astype(np.uint8), cv2.COLOR_HSV2BGR)

    if return_mask:
        # Convert soft mask [0,1] to uint8 [0,255] grayscale
        mask_img = (mask * 255).clip(0, 255).astype(np.uint8)
        return result, mask_img
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",       required=True)
    parser.add_argument("--output",      required=True)
    parser.add_argument("--params",      default="ca_params_1.json")
    parser.add_argument("--export_mask", action="store_true",
                        help="Export purple-fringe mask as grayscale PNG alongside corrected images")
    parser.add_argument("--mask_dir",    default=None,
                        help="Directory for mask output (default: <output>_mask)")
    args = parser.parse_args()

    with open(args.params) as f:
        params = json.load(f)
    print("Parameters:", json.dumps(params, indent=2))

    in_dir  = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_dir = None
    if args.export_mask:
        mask_dir = Path(args.mask_dir) if args.mask_dir else Path(str(args.output) + "_mask")
        mask_dir.mkdir(parents=True, exist_ok=True)
        print(f"Mask output: {mask_dir}")

    exts = {".jpg", ".jpeg", ".png"}
    files = sorted(f for f in in_dir.rglob("*") if f.suffix.lower() in exts)
    print(f"Found {len(files)} images")

    for fpath in tqdm(files):
        rel = fpath.relative_to(in_dir)
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(fpath))
        if img is None:
            tqdm.write(f"Warning: cannot read {rel}, skipping")
            continue

        if mask_dir is not None:
            result, mask_img = process(img, params, return_mask=True)
            mask_path = mask_dir / rel.with_suffix(".png")
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(mask_path), mask_img)
        else:
            result = process(img, params)

        cv2.imwrite(str(out_path), result, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"Done. Output: {out_dir}")

if __name__ == "__main__":
    main()
