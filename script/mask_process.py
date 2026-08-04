#!/usr/bin/env python3
"""Two independent modes, selected by the RUN_MODE macro below:

ANNOTATE (interactive labeling, unchanged from before)
  Zero out the alpha channel below a line (2 points) or above two lines
  (3 points) on a single reference image. Shows the image in a window and
  lets the user click points with the mouse; which mode is used is
  controlled by the MODE macro:
    - "2POINT_BELOW": click p1, p2. Pixels below the line p1-p2 are masked.
    - "3POINT_ABOVE": click p1, p2, p3. Line 1 = p1-p2, line 2 = p2-p3.
                      Pixels above BOTH lines are masked.
  Output is saved as PNG (JPG has no alpha channel) next to the reference
  image, e.g. <DATASET_DIR>/images/cameraXX_ref/<frame>_<sub>_masked.png.

BATCH_MASK (non-interactive batch processing)
  Reads every "<frame>_<sub>_masked.png" annotation produced by ANNOTATE
  mode under <DATASET_DIR>/images/cameraXX_ref/ (the frame number is not
  important, only <sub> is), and uses its alpha channel (alpha == 0) as a
  region to blacken. That region is then blacked out on every image
  images_ori/cameraXX/*_<sub>.jpg (any frame number), and the result is
  written to images/cameraXX/*_<sub>.jpg, leaving images_ori untouched.
  Sub-cameras with no ref annotation are copied through unchanged.
"""

import os
import re
import shutil

import matplotlib
# Force a real interactive backend: PyCharm's built-in "Plots" tool window
# renders static images and doesn't forward mouse clicks to ginput().
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ── Edit these before running ────────────────────────────────────────────────
RUN_MODE = "ANNOTATE"   # "ANNOTATE" or "BATCH_MASK" -- pick exactly one.

# --- ANNOTATE mode settings ---
MODE = "2POINT_BELOW"   # "2POINT_BELOW" or "3POINT_ABOVE"
IMAGE_PATH = "/home/yiinqiang/SMBC/data/Wakasugi_FJD_jb1_2026-06-09-16-06-58/images/camera01_ref/002000_2.jpg"
OUTPUT_PATH = None      # None -> "<image_stem>_masked.png"

# --- BATCH_MASK mode settings ---
DATASET_DIR = "/home/yiinqiang/SMBC/data/Wakasugi_FJD_jb1_2026-06-09-16-06-58"
# Each job reads <DATASET_DIR>/<ref_root>/cameraXX_ref/<frame>_<sub>_masked.png
# annotations, applies them to <DATASET_DIR>/<ori_root>/cameraXX/<frame>_<sub>.jpg,
# and writes the result to <DATASET_DIR>/<out_root>/cameraXX/<frame>_<sub>.jpg.
BATCH_JOBS = [
    # (ref_root, ori_root, out_root)
    ("mask_ref", "images_ori", "images"),
    ("mask_ref", "mask_ori", "mask"),
]

MASKED_FILE_RE = re.compile(r"^(\d+)_(\d+)_masked\.png$")
ORI_FILE_RE = re.compile(r"^\d+_(\d+)\.jpg$")


# ── ANNOTATE mode ─────────────────────────────────────────────────────────────
def pick_points(image_path: str, n: int, title: str):
    """Show the image and let the user click n points; return them in
    original-image pixel coordinates."""
    img = np.array(Image.open(image_path).convert("RGB"))

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(title)
    pts = fig.ginput(n, timeout=0)
    plt.close(fig)

    if len(pts) != n:
        raise RuntimeError(f"Need exactly {n} points to be clicked.")
    return pts


def half_plane_mask(height: int, width: int, p1, p2, above: bool) -> np.ndarray:
    """Return a (H, W) bool mask, True for pixels above/below the line through p1, p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1

    yy, xx = np.mgrid[0:height, 0:width]
    cross = dx * (yy - y1) - dy * (xx - x1)

    # Reference point clearly above/below both endpoints, used to pin down
    # which side of the line corresponds to "above" (robust to line orientation).
    ref_x = (x1 + x2) / 2.0
    ref_y = (min(y1, y2) - 1.0) if above else (max(y1, y2) + 1.0)
    ref_cross = dx * (ref_y - y1) - dy * (ref_x - x1)

    if ref_cross >= 0:
        return cross >= 0
    return cross <= 0


def above_both_lines_mask(height: int, width: int, p1, p2, p3) -> np.ndarray:
    """True for pixels above BOTH line p1-p2 and line p2-p3."""
    mask1 = half_plane_mask(height, width, p1, p2, above=True)
    mask2 = half_plane_mask(height, width, p2, p3, above=True)
    return mask1 & mask2


def apply_alpha_mask(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Set alpha to 0 wherever mask is True."""
    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    arr[mask, 3] = 0
    return Image.fromarray(arr, mode="RGBA")


def run_annotate() -> None:
    output_path = OUTPUT_PATH
    if output_path is None:
        stem = os.path.splitext(os.path.abspath(IMAGE_PATH))[0]
        output_path = stem + "_masked.png"

    img = Image.open(IMAGE_PATH)

    if MODE == "2POINT_BELOW":
        p1, p2 = pick_points(IMAGE_PATH, 2,
                              "Click p1 (top-left) then p2 (bottom-right)")
        print(f"Point 1: {p1}, Point 2: {p2}")
        mask = half_plane_mask(img.height, img.width, p1, p2, above=False)
    elif MODE == "3POINT_ABOVE":
        p1, p2, p3 = pick_points(IMAGE_PATH, 3,
                                  "Click p1, p2 (line 1 = p1-p2), then p3 (line 2 = p2-p3)")
        print(f"Point 1: {p1}, Point 2: {p2}, Point 3: {p3}")
        mask = above_both_lines_mask(img.height, img.width, p1, p2, p3)
    else:
        raise ValueError(f"Unknown MODE: {MODE!r}")

    out = apply_alpha_mask(img, mask)
    out.save(output_path)
    print(f"Saved → {output_path}")


# ── BATCH_MASK mode ───────────────────────────────────────────────────────────
def find_camera_masks(ref_dir: str) -> dict:
    """Scan a cameraXX_ref directory for '<frame>_<sub>_masked.png' files and
    return {sub_id: mask_png_path}.

    Raises if the same sub-camera id has more than one *_masked.png in this
    directory, since it would be ambiguous which annotation to use.
    """
    masks_by_sub = {}
    for name in sorted(os.listdir(ref_dir)):
        m = MASKED_FILE_RE.match(name)
        if not m:
            continue
        sub_id = m.group(2)
        path = os.path.join(ref_dir, name)
        if sub_id in masks_by_sub:
            raise RuntimeError(
                f"Multiple *_masked.png found for sub-camera '{sub_id}' in {ref_dir}: "
                f"{masks_by_sub[sub_id]} and {path}"
            )
        masks_by_sub[sub_id] = path
    return masks_by_sub


def alpha_hole_mask(mask_png_path: str) -> np.ndarray:
    """Return a (H, W) bool mask, True wherever the annotation's alpha == 0."""
    rgba = np.array(Image.open(mask_png_path).convert("RGBA"))
    return rgba[:, :, 3] == 0


def blacken_and_save(src_path: str, mask: np.ndarray, dst_path: str) -> None:
    arr = np.array(Image.open(src_path).convert("RGB"))
    if arr.shape[:2] != mask.shape:
        raise RuntimeError(
            f"Size mismatch between {src_path} {arr.shape[:2]} and mask {mask.shape}"
        )
    arr[mask] = 0
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(dst_path, quality=95)


def process_camera(ref_dir: str, ori_root: str, out_root: str) -> None:
    camera_name = os.path.basename(ref_dir.rstrip("/"))[: -len("_ref")]

    ori_dir = os.path.join(ori_root, camera_name)
    if not os.path.isdir(ori_dir):
        print(f"[skip] {camera_name}: no such directory {ori_dir}")
        return

    masks_by_sub = find_camera_masks(ref_dir)
    if not masks_by_sub:
        print(f"[skip] {camera_name}: no *_masked.png annotations found in {ref_dir}")
        return

    out_dir = os.path.join(out_root, camera_name)
    ori_names = sorted(os.listdir(ori_dir))
    unhandled_names = set(ori_names)

    for sub_id, mask_path in masks_by_sub.items():
        mask = alpha_hole_mask(mask_path)
        pattern = re.compile(rf"^\d+_{re.escape(sub_id)}\.jpg$")
        matched = [n for n in ori_names if pattern.match(n)]
        if not matched:
            print(f"[warn] {camera_name}: no images_ori files matched sub-camera '{sub_id}'")
            continue
        for name in matched:
            blacken_and_save(os.path.join(ori_dir, name), mask, os.path.join(out_dir, name))
        unhandled_names.difference_update(matched)
        print(f"[ok] {camera_name} sub-camera {sub_id}: masked {len(matched)} images -> {out_dir}")

    # Sub-cameras with no ref annotation are copied through unchanged.
    copied_by_sub = {}
    for name in sorted(unhandled_names):
        m = ORI_FILE_RE.match(name)
        if not m:
            continue
        os.makedirs(out_dir, exist_ok=True)
        shutil.copy2(os.path.join(ori_dir, name), os.path.join(out_dir, name))
        copied_by_sub[m.group(1)] = copied_by_sub.get(m.group(1), 0) + 1
    for sub_id, count in sorted(copied_by_sub.items()):
        print(f"[copy] {camera_name} sub-camera {sub_id}: no annotation, copied {count} images unchanged -> {out_dir}")


def run_batch_mask_job(dataset_dir: str, ref_root_name: str, ori_root_name: str, out_root_name: str) -> None:
    ref_root = os.path.join(dataset_dir, ref_root_name)
    ori_root = os.path.join(dataset_dir, ori_root_name)
    out_root = os.path.join(dataset_dir, out_root_name)

    ref_dirs = sorted(
        os.path.join(ref_root, d)
        for d in os.listdir(ref_root)
        if d.endswith("_ref") and os.path.isdir(os.path.join(ref_root, d))
    )
    if not ref_dirs:
        raise RuntimeError(f"No *_ref directories found under {ref_root}")

    for ref_dir in ref_dirs:
        process_camera(ref_dir, ori_root, out_root)


def run_batch_mask(dataset_dir: str) -> None:
    for ref_root_name, ori_root_name, out_root_name in BATCH_JOBS:
        print(f"[job] {ref_root_name} -> {out_root_name}")
        run_batch_mask_job(dataset_dir, ref_root_name, ori_root_name, out_root_name)


if __name__ == "__main__":
    if RUN_MODE == "ANNOTATE":
        run_annotate()
    elif RUN_MODE == "BATCH_MASK":
        run_batch_mask(DATASET_DIR)
    else:
        raise ValueError(f"Unknown RUN_MODE: {RUN_MODE!r}")
