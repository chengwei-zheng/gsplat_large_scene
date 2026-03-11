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

    # Merge two point clouds
    python script/convert_las.py --merge file1.txt file2.txt --output_dir data/merged

    # Add sky hemisphere
    python script/convert_las.py --input data/points3D.txt --add_sky --output_dir data/with_sky

    # Add sky hemisphere with custom radius and ratio
    python script/convert_las.py --input data/points3D.txt --add_sky --sky_radius 100 --sky_ratio 0.3

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


def generate_sky_points(points, sky_radius=50.0, ratio=0.5):
    """Generate sky point cloud as a hemisphere above the scene.

    The hemisphere is centered at (x_center, y_center, 0) where x_center and
    y_center are the center of the original point cloud's x and y range.

    Args:
        points: (N, 3) original point cloud
        sky_radius: radius of the hemisphere (default: 50.0)
        ratio: ratio of sky points to original points (default: 0.5)

    Returns:
        sky_points: (M, 3) sky point coordinates on hemisphere
        sky_colors: (M, 3) sky point colors (white-blue range)
    """
    n_sky = int(len(points) * ratio)

    # Get center of x, y from original points
    x_center = (points[:, 0].min() + points[:, 0].max()) / 2
    y_center = (points[:, 1].min() + points[:, 1].max()) / 2

    # Generate random points on upper hemisphere using spherical coordinates
    # theta: azimuthal angle [0, 2*pi]
    # phi: polar angle [0, pi/2] for upper hemisphere (0=top, pi/2=horizon)
    theta = np.random.uniform(0, 2 * np.pi, n_sky)
    # For uniform distribution on hemisphere: phi = arccos(1 - u) where u in [0, 1]
    # This gives phi in [0, pi/2]
    u = np.random.uniform(0, 1, n_sky)
    phi = np.arccos(1 - u)

    # Convert to Cartesian coordinates
    sky_x = x_center + sky_radius * np.sin(phi) * np.cos(theta)
    sky_y = y_center + sky_radius * np.sin(phi) * np.sin(theta)
    sky_z = sky_radius * np.cos(phi)  # Always positive for upper hemisphere

    sky_points = np.stack([sky_x, sky_y, sky_z], axis=-1).astype(np.float64)

    # Random colors in white-blue range
    # White: (255, 255, 255), Blue: (135, 206, 235) sky blue
    # Lower points (near horizon) more blue, higher points more white
    t = sky_z / sky_radius  # 0 at horizon, 1 at top
    t = 1 - t  # Invert: more blue at top, more white at horizon (optional)
    t = np.random.uniform(0, 1, n_sky)  # Or just random
    r = (255 * (1 - t) + 135 * t).astype(np.uint8)
    g = (255 * (1 - t) + 206 * t).astype(np.uint8)
    b = np.full(n_sky, 255, dtype=np.uint8)  # Always high blue

    sky_colors = np.stack([r, g, b], axis=-1)

    print(f"Generated {n_sky} sky points as hemisphere "
          f"(center: [{x_center:.2f}, {y_center:.2f}, 0], radius: {sky_radius})")

    return sky_points, sky_colors


def read_point_cloud(file_path):
    """Read point cloud from LAS, PLY, or points3D.txt file.

    Returns:
        points: (N, 3) float64 array of xyz coordinates
        colors: (N, 3) uint8 array of RGB colors
    """
    ext = os.path.splitext(file_path)[1].lower()
    name = os.path.basename(file_path).lower()

    if ext in [".las", ".laz"]:
        return read_las(file_path)
    elif ext == ".txt" or name.startswith("points3d"):
        return read_points3d_txt(file_path)
    else:
        raise ValueError(f"Unsupported format: {ext}. Supported: .las, .laz, .txt (points3D)")


def merge_point_clouds(file_paths):
    """Merge multiple point cloud files.

    Args:
        file_paths: List of paths to point cloud files

    Returns:
        points: (N, 3) float64 array of merged xyz coordinates
        colors: (N, 3) uint8 array of merged RGB colors
    """
    all_points = []
    all_colors = []

    for file_path in file_paths:
        points, colors = read_point_cloud(file_path)
        all_points.append(points)
        all_colors.append(colors)

    merged_points = np.concatenate(all_points, axis=0)
    merged_colors = np.concatenate(all_colors, axis=0)

    print(f"Merged {len(file_paths)} files -> {len(merged_points)} total points")
    return merged_points, merged_colors


def main():
    p = argparse.ArgumentParser(
        description="Convert point cloud between LAS, PLY, and COLMAP points3D.txt formats"
    )
    p.add_argument("--input", type=str, default=None,
                   help="Path to input file (LAS or points3D.txt)")
    p.add_argument("--merge", type=str, nargs="+", default=None,
                   help="Merge multiple point cloud files (LAS, PLY, or points3D.txt)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory (default: same as input file)")
    p.add_argument("--export_ply", action=argparse.BooleanOptionalAction, default=True,
                   help="Export PLY file (default: True)")
    p.add_argument("--export_points3d", action=argparse.BooleanOptionalAction, default=True,
                   help="Export COLMAP points3D.txt (default: True)")
    p.add_argument("--subsample", type=int, default=None,
                   help="Subsample points (keep every N-th point)")
    p.add_argument("--add_sky", action="store_true", default=False,
                   help="Add sky points as a hemisphere above the scene")
    p.add_argument("--sky_radius", type=float, default=50.0,
                   help="Radius of the sky hemisphere (default: 50.0)")
    p.add_argument("--sky_ratio", type=float, default=0.5,
                   help="Ratio of sky points to original points (default: 0.5)")
    args = p.parse_args()

    # Check input arguments
    if args.input is None and args.merge is None:
        p.error("Either --input or --merge is required")
    if args.input is not None and args.merge is not None:
        p.error("Cannot use both --input and --merge")

    # Merge mode
    if args.merge is not None:
        if len(args.merge) < 2:
            p.error("--merge requires at least 2 files")

        if args.output_dir is None:
            args.output_dir = os.path.dirname(os.path.abspath(args.merge[0]))
        os.makedirs(args.output_dir, exist_ok=True)

        points, colors = merge_point_clouds(args.merge)
        base_name = "merged"
    else:
        # Single input mode
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
                print("Input is points3D.txt, skipping points3D.txt     export")
                args.export_points3d = False
        else:
            raise ValueError(f"Unsupported input format: {input_ext}. Supported: .las, .laz, .txt (points3D)")

    # Subsample if requested
    if args.subsample is not None and args.subsample > 1:
        indices = np.arange(0, len(points), args.subsample)
        points = points[indices]
        colors = colors[indices]
        print(f"Subsampled to {len(points)} points (every {args.subsample}-th point)")

    # Add sky points if requested
    if args.add_sky:
        sky_points, sky_colors = generate_sky_points(
            points, sky_radius=args.sky_radius, ratio=args.sky_ratio
        )
        points = np.concatenate([points, sky_points], axis=0)
        colors = np.concatenate([colors, sky_colors], axis=0)
        print(f"Total points after adding sky: {len(points)}")

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
