#!/usr/bin/env python3
"""Convert point cloud between LAS, PLY, and COLMAP points3D.txt formats.

Usage:
    # LAS -> PLY + points3D.txt
    python script/convert_las.py --input data/scene.las --output_dir data/converted

    # LAS -> PLY only
    python script/convert_las.py --input data/scene.las --output_dir data/converted --no-export_points3d

    # LAS -> points3D.txt only
    python script/convert_las.py --input data/scene.las --output_dir data/converted --no-export_ply

    # points3D.txt -> PLY
    python script/convert_las.py --input sparse/0/points3D.txt --output_dir data/converted

Requirements:
    pip install laspy  (for LAS files)
"""

import argparse
import os

import numpy as np


def read_las(las_path):
    """Read LAS file and return xyz coordinates and RGB colors.

    Returns:
        points: (N, 3) float64 array of xyz coordinates
        colors: (N, 3) uint8 array of RGB colors, or None if not available
    """
    try:
        import laspy
    except ImportError:
        raise ImportError("laspy is required. Install with: pip install laspy")

    print(f"Reading LAS file: {las_path}")
    las = laspy.read(las_path)

    # Extract xyz coordinates
    points = np.stack([las.x, las.y, las.z], axis=-1).astype(np.float64)
    print(f"Loaded {len(points)} points")

    # Try to extract RGB colors
    colors = None
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        r = np.array(las.red)
        g = np.array(las.green)
        b = np.array(las.blue)

        # LAS RGB values are typically 16-bit (0-65535), convert to 8-bit
        if r.max() > 255 or g.max() > 255 or b.max() > 255:
            r = (r / 256).astype(np.uint8)
            g = (g / 256).astype(np.uint8)
            b = (b / 256).astype(np.uint8)
        else:
            r = r.astype(np.uint8)
            g = g.astype(np.uint8)
            b = b.astype(np.uint8)

        colors = np.stack([r, g, b], axis=-1)
        print("RGB colors found")
    else:
        print("No RGB colors in LAS file, using gray (128, 128, 128)")
        colors = np.full((len(points), 3), 128, dtype=np.uint8)

    return points, colors


def read_points3d_txt(txt_path):
    """Read COLMAP points3D.txt and return xyz coordinates and RGB colors.

    Returns:
        points: (N, 3) float64 array of xyz coordinates
        colors: (N, 3) uint8 array of RGB colors
    """
    print(f"Reading points3D.txt: {txt_path}")

    points_list = []
    colors_list = []

    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line.startswith("#") or len(line) == 0:
                continue

            parts = line.split()
            if len(parts) < 7:
                continue

            # Format: POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            r, g, b = int(parts[4]), int(parts[5]), int(parts[6])

            points_list.append([x, y, z])
            colors_list.append([r, g, b])

    points = np.array(points_list, dtype=np.float64)
    colors = np.array(colors_list, dtype=np.uint8)

    print(f"Loaded {len(points)} points")
    return points, colors


def export_ply(points, colors, output_path):
    """Export points and colors to binary PLY file."""
    n_points = len(points)
    points = points.astype(np.float32)

    with open(output_path, "wb") as f:
        # PLY header
        header = f"""ply
format binary_little_endian 1.0
element vertex {n_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
        f.write(header.encode("ascii"))

        # Write binary data: xyz (float32) + rgb (uint8)
        for i in range(n_points):
            f.write(points[i].tobytes())  # 3 x float32
            f.write(colors[i].tobytes())  # 3 x uint8

    print(f"PLY exported to {output_path} ({n_points} points)")


def export_points3d_txt(points, colors, output_path):
    """Export points and colors to COLMAP points3D.txt format."""
    n_points = len(points)

    with open(output_path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {n_points}, converted from LAS\n")

        for i in range(n_points):
            point_id = i + 1
            x, y, z = points[i]
            r, g, b = colors[i]
            error = 0.0  # placeholder
            # Empty track (no image observations)
            f.write(f"{point_id} {x:.10f} {y:.10f} {z:.10f} {r} {g} {b} {error:.6f}\n")

    print(f"points3D.txt exported to {output_path} ({n_points} points)")


def main():
    p = argparse.ArgumentParser(
        description="Convert point cloud between LAS, PLY, and COLMAP points3D.txt formats"
    )
    p.add_argument("--input", type=str, required=True,
                   help="Path to input file (LAS or points3D.txt)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory (default: same as input file)")
    p.add_argument("--export_ply", action=argparse.BooleanOptionalAction, default=True,
                   help="Export PLY file (default: True)")
    p.add_argument("--export_points3d", action=argparse.BooleanOptionalAction, default=True,
                   help="Export COLMAP points3D.txt (default: True)")
    p.add_argument("--subsample", type=int, default=None,
                   help="Subsample points (keep every N-th point)")
    args = p.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(args.input))

    os.makedirs(args.output_dir, exist_ok=True)

    # Detect input file type and read
    input_ext = os.path.splitext(args.input)[1].lower()
    input_name = os.path.basename(args.input).lower()

    if input_ext in [".las", ".laz"]:
        points, colors = read_las(args.input)
        base_name = os.path.splitext(os.path.basename(args.input))[0]
    elif input_ext == ".txt" or input_name.startswith("points3d"):
        points, colors = read_points3d_txt(args.input)
        base_name = "points3D"
        # If input is points3D.txt, skip exporting points3D.txt by default
        if args.export_points3d and input_name.startswith("points3d"):
            print("Input is points3D.txt, skipping points3D.txt export")
            args.export_points3d = False
    else:
        raise ValueError(f"Unsupported input format: {input_ext}. Supported: .las, .laz, .txt (points3D)")

    # Subsample if requested
    if args.subsample is not None and args.subsample > 1:
        indices = np.arange(0, len(points), args.subsample)
        points = points[indices]
        colors = colors[indices]
        print(f"Subsampled to {len(points)} points (every {args.subsample}-th point)")

    # Export PLY
    if args.export_ply:
        ply_path = os.path.join(args.output_dir, f"{base_name}.ply")
        export_ply(points, colors, ply_path)

    # Export points3D.txt
    if args.export_points3d:
        points3d_path = os.path.join(args.output_dir, "points3D.txt")
        export_points3d_txt(points, colors, points3d_path)

    print("Done.")


if __name__ == "__main__":
    main()
