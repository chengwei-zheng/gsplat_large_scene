"""
Split COLMAP sparse/0 into 4 quadrant blocks for block-wise 3DGS training.

Splits cameras by mean XY position into 4 quadrants. cameras.bin and
points3D.bin are shared (symlinked) across all blocks.

Output structure:
    output_dir/
        cameras.bin
        points3D.bin
        sparse_00/   x < mean_x, y < mean_y
            images.bin
            cameras.bin  -> ../cameras.bin
            points3D.bin -> ../points3D.bin
        sparse_01/   x < mean_x, y >= mean_y
        sparse_10/   x >= mean_x, y < mean_y
        sparse_11/   x >= mean_x, y >= mean_y
        split_info.json

Usage:
    python split_colmap.py \\
        --input  /path/to/sparse/0 \\
        --output /path/to/output
"""

import argparse
import array as array_module
import json
import os
import shutil
import struct
import numpy as np
from pycolmap import SceneManager


# ---------------------------------------------------------------------------
# Binary images.bin filter
# ---------------------------------------------------------------------------

# COLMAP images.bin format per image:
#   image_id       : uint32
#   qw qx qy qz    : 4x float64
#   tx ty tz        : 3x float64
#   camera_id      : uint32
#   name           : null-terminated UTF-8 string
#   num_points2D   : uint64
#   points2D       : num_points2D * (x:f64, y:f64, point3D_id:u64-as-f64)

_IMAGE_STRUCT = struct.Struct("<I 4d 3d I")


def filter_images_bin(input_path, output_path, keep_ids):
    """Read images.bin and write a filtered copy containing only keep_ids."""
    keep_ids = set(keep_ids)

    # ---- pass 1: read all records ----------------------------------------
    records = []  # list of (image_id, fixed_bytes, name_bytes, num_pts, pts_bytes)
    with open(input_path, "rb") as f:
        num_images = struct.unpack("Q", f.read(8))[0]
        for _ in range(num_images):
            fixed = f.read(_IMAGE_STRUCT.size)
            image_id = _IMAGE_STRUCT.unpack(fixed)[0]

            name_bytes = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_bytes += c

            num_pts = struct.unpack("Q", f.read(8))[0]
            pts_bytes = f.read(num_pts * 3 * 8)  # 3 x float64 per point

            if image_id in keep_ids:
                records.append((image_id, fixed, name_bytes, num_pts, pts_bytes))

    # ---- pass 2: write filtered file -------------------------------------
    with open(output_path, "wb") as f:
        f.write(struct.pack("Q", len(records)))
        for _, fixed, name_bytes, num_pts, pts_bytes in sorted(records):
            f.write(fixed)
            f.write(name_bytes + b"\x00")
            f.write(struct.pack("Q", num_pts))
            f.write(pts_bytes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Split COLMAP sparse/0 into 4 XY-quadrant blocks"
    )
    parser.add_argument("--input", required=True, help="Path to COLMAP sparse/0 directory")
    parser.add_argument("--output", required=True, help="Path to output directory")
    parser.add_argument("--pad", type=float, default=2.0,
                        help="Overlap padding in COLMAP units at each split boundary (default: 2.0)")
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output

    # ------------------------------------------------------------------
    # Load camera positions via SceneManager (reads .bin automatically)
    # ------------------------------------------------------------------
    print(f"Loading COLMAP data from {input_dir} ...")
    manager = SceneManager(input_dir)
    manager.load_cameras()
    manager.load_images()

    imdata = manager.images
    print(f"  {len(imdata)} images loaded")

    centers = {iid: im.C() for iid, im in imdata.items()}

    all_xy = np.array([c[:2] for c in centers.values()])
    mean_x, mean_y = float(all_xy[:, 0].mean()), float(all_xy[:, 1].mean())
    print(f"  Scene XY center: ({mean_x:.3f}, {mean_y:.3f})")

    # ------------------------------------------------------------------
    # Assign images to quadrants (with overlap padding)
    # first digit  = x (0: x < mean_x,  1: x >= mean_x)
    # second digit = y (0: y < mean_y,  1: y >= mean_y)
    # Each block is extended by `pad` into the neighbouring side so that
    # cameras within `pad` of the split line appear in both blocks.
    # ------------------------------------------------------------------
    pad = args.pad
    print(f"  Split pad: {pad} (overlap at boundary)")
    quadrants = {"00": [], "01": [], "10": [], "11": []}
    for iid, center in centers.items():
        cx, cy = center[0], center[1]
        if cx < mean_x + pad:
            if cy < mean_y + pad:
                quadrants["00"].append(iid)
            if cy >= mean_y - pad:
                quadrants["01"].append(iid)
        if cx >= mean_x - pad:
            if cy < mean_y + pad:
                quadrants["10"].append(iid)
            if cy >= mean_y - pad:
                quadrants["11"].append(iid)

    for key, ids in quadrants.items():
        print(f"  sparse_{key}: {len(ids)} images")

    # ------------------------------------------------------------------
    # Locate source images.bin
    # ------------------------------------------------------------------
    images_bin_src = os.path.join(input_dir, "images.bin")
    if not os.path.exists(images_bin_src):
        raise FileNotFoundError(f"images.bin not found in {input_dir}")

    # ------------------------------------------------------------------
    # Create output directory and copy shared files
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    for filename in ["cameras.bin", "points3D.bin"]:
        src = os.path.join(input_dir, filename)
        dst = os.path.join(output_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Copied {filename}")
        else:
            print(f"  Warning: {filename} not found in input, skipping")

    # ------------------------------------------------------------------
    # Create per-quadrant directories
    # ------------------------------------------------------------------
    for key, image_ids in quadrants.items():
        quad_dir = os.path.join(output_dir, f"sparse_{key}")
        os.makedirs(quad_dir, exist_ok=True)

        # Write filtered images.bin
        filter_images_bin(
            images_bin_src,
            os.path.join(quad_dir, "images.bin"),
            image_ids,
        )

        # Symlink shared files
        for filename in ["cameras.bin", "points3D.bin"]:
            link_path = os.path.join(quad_dir, filename)
            if os.path.lexists(link_path):
                os.remove(link_path)
            os.symlink(os.path.join("..", filename), link_path)

        print(f"  Created sparse_{key}/ ({len(image_ids)} images)")

    # ------------------------------------------------------------------
    # Write split_info.json
    # ------------------------------------------------------------------
    all_centers_arr = np.array(list(centers.values()))
    log = {
        "input_dir": os.path.abspath(input_dir),
        "output_dir": os.path.abspath(output_dir),
        "total_images": len(imdata),
        "split_center": {"x": mean_x, "y": mean_y},
        "split_pad": pad,
        "scene_xy_min": {
            "x": float(all_centers_arr[:, 0].min()),
            "y": float(all_centers_arr[:, 1].min()),
        },
        "scene_xy_max": {
            "x": float(all_centers_arr[:, 0].max()),
            "y": float(all_centers_arr[:, 1].max()),
        },
        "quadrants": {},
    }
    for key, image_ids in quadrants.items():
        if not image_ids:
            continue
        q_centers = np.array([centers[iid] for iid in image_ids])
        log["quadrants"][f"sparse_{key}"] = {
            "num_images": len(image_ids),
            "x_range": [float(q_centers[:, 0].min()), float(q_centers[:, 0].max())],
            "y_range": [float(q_centers[:, 1].min()), float(q_centers[:, 1].max())],
            "z_range": [float(q_centers[:, 2].min()), float(q_centers[:, 2].max())],
            "image_ids": sorted(image_ids),
        }

    with open(os.path.join(output_dir, "split_info.json"), "w") as f:
        json.dump(log, f, indent=2)
    print("  Wrote split_info.json")

    print(f"\nDone. Output: {output_dir}")
    print("Train each block with:  --sparse_folder sparse_blocks/sparse_XX")


if __name__ == "__main__":
    main()
