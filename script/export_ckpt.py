#!/usr/bin/env python3
"""Export point clouds and COLMAP poses from a gsplat checkpoint (.pt file).

Usage:
    # Export standard 3DGS splat PLY (for Gaussian splatting viewers, normalized space)
    python script/export_ckpt.py --ckpt results/xxx/ckpts/ckpt_merged.pt --export_splat_ply

    # Export COLMAP points3D.txt for re-initialization (metric coordinates)
    python script/export_ckpt.py --ckpt results/xxx/ckpts/ckpt_merged.pt --export_points3d

    # Export camera poses
    python script/export_ckpt.py --ckpt results/xxx/ckpts/ckpt_merged.pt --export_poses --data_dir data/xxx

    # Export point cloud for treeiso input (opacity-filtered, xyz-only)
    python script/export_ckpt.py --ckpt results/xxx/ckpts/ckpt_merged.pt --export_treeiso_points

    # Reset checkpoint step to 0 for re-training
    python script/export_ckpt.py --ckpt results/xxx/ckpts/ckpt_merged.pt --reset_step
"""

import argparse
import os
import shutil
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(__file__))

from datasets.colmap import Parser
from utils import CameraOptModule

from gsplat import export_splats


def _inverse_transform(means, transform):
    """Apply inverse of the stored normalization transform to get metric coordinates."""
    if transform is None:
        print("Warning: checkpoint has no 'transform'; exporting normalized coordinates")
        return means
    T = transform.numpy() if isinstance(transform, torch.Tensor) else np.array(transform)
    T_inv = np.linalg.inv(T.astype(np.float64))
    return means @ T_inv[:3, :3].T + T_inv[:3, 3]


def _sh0_to_rgb(sh0):
    """Convert SH DC coefficients to uint8 RGB."""
    C0 = 0.28209479177387814
    if isinstance(sh0, torch.Tensor):
        sh0 = sh0.numpy()
    rgb = sh0.squeeze(1) * C0 + 0.5  # (N, 3), range [0, 1]
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def do_export_splat_ply(ckpt, output_dir):
    """Export splats as a standard 3DGS PLY file (for Gaussian splatting viewers).

    Coordinates stay in normalized space. Scales and opacities are kept in
    log/logit space; quaternions are not normalized.
    """
    splats = ckpt["splats"]

    if "sh0" not in splats:
        print("Warning: checkpoint uses appearance model (no sh0/shN). Skipping PLY export.")
        return

    os.makedirs(output_dir, exist_ok=True)
    ply_path = os.path.join(output_dir, "point_cloud.ply")

    export_splats(
        means=splats["means"],
        scales=splats["scales"],
        quats=splats["quats"],
        opacities=splats["opacities"],
        sh0=splats["sh0"],
        shN=splats["shN"],
        format="ply",
        save_to=ply_path,
    )
    print(f"Splat PLY exported to {ply_path} ({len(splats['means']):,} Gaussians)")


def do_export_points3d(ckpt, output_dir):
    """Export Gaussian means + colors as COLMAP points3D.txt in metric coordinates.

    Applies the inverse of the stored normalization transform so coordinates are
    in the original COLMAP world frame (metres). Colors are derived from SH DC.
    """
    splats = ckpt["splats"]
    means = splats["means"]

    if isinstance(means, torch.Tensor):
        means = means.numpy()
    means = _inverse_transform(means.astype(np.float64), ckpt.get("transform"))

    if "sh0" not in splats:
        print("Warning: checkpoint uses appearance model (no sh0). Using gray colors.")
        colors = np.full((len(means), 3), 128, dtype=np.uint8)
    else:
        colors = _sh0_to_rgb(splats["sh0"])

    n_points = len(means)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "points3D.txt")

    with open(output_path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {n_points}, exported from gsplat checkpoint\n")
        for i in range(n_points):
            x, y, z = means[i]
            r, g, b = colors[i]
            f.write(f"{i+1} {x:.10f} {y:.10f} {z:.10f} {r} {g} {b} 0.000000\n")

    print(f"points3D.txt exported to {output_path} ({n_points:,} points)")


def _compute_obb_2d(points_xy):
    """Compute the minimum-area oriented bounding rectangle of a 2D point set.

    Uses rotating calipers over the convex hull edges.
    Returns (angle_rad, min_u, max_u, min_v, max_v) where u/v are axes of the
    rotated frame aligned with the rectangle.
    """
    from scipy.spatial import ConvexHull
    hull = ConvexHull(points_xy)
    hull_pts = points_xy[hull.vertices]
    n = len(hull_pts)

    min_area = np.inf
    best = None
    for i in range(n):
        edge = hull_pts[(i + 1) % n] - hull_pts[i]
        angle = np.arctan2(edge[1], edge[0])
        c, s = np.cos(-angle), np.sin(-angle)
        rot = hull_pts @ np.array([[c, -s], [s, c]]).T
        min_u, max_u = rot[:, 0].min(), rot[:, 0].max()
        min_v, max_v = rot[:, 1].min(), rot[:, 1].max()
        area = (max_u - min_u) * (max_v - min_v)
        if area < min_area:
            min_area = area
            best = (angle, min_u, max_u, min_v, max_v)
    return best


def _filter_by_obb_2d(means_world, bbox_ply_path, margin=0.0):
    """Return boolean mask (N,) for means_world points inside the 2D OBB from bbox_ply_path.

    The OBB is the minimum-area bounding rectangle of the XY projection of the
    bbox PLY. Z is ignored — the constraint is purely in XY.
    margin: shrink each side of the rectangle inward by this many metres.
    """
    from convert_las import read_ply as _read_ply
    bbox_pts, _ = _read_ply(bbox_ply_path)  # (M, 3) float64
    print(f"  bbox_ply XY range: x=[{bbox_pts[:,0].min():.2f}, {bbox_pts[:,0].max():.2f}]  "
          f"y=[{bbox_pts[:,1].min():.2f}, {bbox_pts[:,1].max():.2f}]  ({len(bbox_pts)} pts)")
    print(f"  means_world XY range: x=[{means_world[:,0].min():.2f}, {means_world[:,0].max():.2f}]  "
          f"y=[{means_world[:,1].min():.2f}, {means_world[:,1].max():.2f}]")

    angle, min_u, max_u, min_v, max_v = _compute_obb_2d(bbox_pts[:, :2])
    print(f"  OBB angle={np.degrees(angle):.2f}°  "
          f"u=[{min_u:.2f}, {max_u:.2f}] ({max_u-min_u:.2f}m)  "
          f"v=[{min_v:.2f}, {max_v:.2f}] ({max_v-min_v:.2f}m)")

    min_u += margin
    max_u -= margin
    min_v += margin
    max_v -= margin
    print(f"  OBB after margin ({margin}m): u=[{min_u:.2f}, {max_u:.2f}]  v=[{min_v:.2f}, {max_v:.2f}]")

    if min_u > max_u or min_v > max_v:
        raise ValueError(
            f"bbox_margin={margin} is too large: OBB collapsed after shrinking "
            f"(u: {max_u-min_u+2*margin:.3f}m wide → {max_u-min_u:.3f}m after margin, "
            f"v: {max_v-min_v+2*margin:.3f}m wide → {max_v-min_v:.3f}m after margin). "
            f"Reduce --bbox_margin."
        )

    c, s = np.cos(-angle), np.sin(-angle)
    rotated = means_world[:, :2] @ np.array([[c, -s], [s, c]]).T
    print(f"  rotated means_world: u=[{rotated[:,0].min():.2f}, {rotated[:,0].max():.2f}]  "
          f"v=[{rotated[:,1].min():.2f}, {rotated[:,1].max():.2f}]")

    inside = (
        (rotated[:, 0] >= min_u) & (rotated[:, 0] <= max_u) &
        (rotated[:, 1] >= min_v) & (rotated[:, 1] <= max_v)
    )
    print(f"OBB filter (margin={margin}m): {(~inside).sum():,} points removed, "
          f"{inside.sum():,} remaining")
    return inside


def do_export_treeiso_points(ckpt, output_dir, bbox_ply=None, bbox_margin=0.0):
    """Export Gaussian means + opacity as a metric-coordinate PLY for treeiso input.

    Removes sky Gaussians and applies optional OBB bbox crop, then writes all
    remaining Gaussians to PLY with sigmoid(opacity) in [0,1] as an extra
    per-point attribute. No opacity threshold is applied at export time —
    filtering by opacity can be done downstream.
    Coordinates are in the original COLMAP world frame (metres).
    """
    splats = ckpt["splats"]
    means = splats["means"]
    opacities = splats["opacities"]

    if isinstance(opacities, torch.Tensor):
        opacities = opacities.numpy()
    if isinstance(means, torch.Tensor):
        means = means.numpy()

    all_idx = np.arange(len(means), dtype=np.int32)

    # Step 1: sky filter (normalized space)
    sky_removed = np.empty(0, dtype=np.int32)
    non_sky_idx = all_idx
    if "sky_hemisphere_center" in ckpt and "sky_depth_min" in ckpt:
        center = ckpt["sky_hemisphere_center"].numpy()
        sky_depth_min = float(ckpt["sky_depth_min"])
        dist = np.linalg.norm(means - center, axis=-1)
        sky_mask = dist > sky_depth_min
        sky_removed = all_idx[sky_mask]
        non_sky_idx = all_idx[~sky_mask]
        print(f"Sky filter: {len(sky_removed):,} sky Gaussians removed, {len(non_sky_idx):,} remaining")
    else:
        print("Warning: checkpoint has no sky parameters; sky Gaussians will not be removed.")

    # Step 2: bbox filter (world/metric space)
    means_non_sky_world = _inverse_transform(means[non_sky_idx].astype(np.float64), ckpt.get("transform"))
    bbox_removed = np.empty(0, dtype=np.int32)
    in_bbox_idx = non_sky_idx
    means_in_bbox_world = means_non_sky_world
    if bbox_ply is not None:
        obb_mask = _filter_by_obb_2d(means_non_sky_world, bbox_ply, margin=bbox_margin)
        bbox_removed = non_sky_idx[~obb_mask]
        in_bbox_idx = non_sky_idx[obb_mask]
        means_in_bbox_world = means_non_sky_world[obb_mask]

    survivor = in_bbox_idx
    means_world = means_in_bbox_world.astype(np.float32)
    opacity_vals = (1.0 / (1.0 + np.exp(-opacities[survivor].astype(np.float64)))).astype(np.float32)
    n_points = len(means_world)
    print(f"Total exported: {n_points:,} (sky: {len(sky_removed):,} removed, bbox: {len(bbox_removed):,} removed)")

    os.makedirs(output_dir, exist_ok=True)

    # Index map: maps each Gaussian to its filter outcome.
    #   survivor[i]  — PLY point i → original Gaussian index (treeiso output[i] maps back here)
    #   sky_removed  — removed by sky hemisphere filter
    #   bbox_removed — not sky, but outside OBB bbox
    # Back-project: gaussian_labels[survivor] = treeiso_labels
    index_map_path = os.path.join(output_dir, "treeiso_input_index_map.npz")
    np.savez(index_map_path,
             survivor=survivor,
             sky_removed=sky_removed,
             bbox_removed=bbox_removed)
    print(f"Saved: {index_map_path} ({n_points:,} survivor, "
          f"{len(sky_removed):,} sky, {len(bbox_removed):,} bbox-removed)")

    ply_path = os.path.join(output_dir, "treeiso_input.ply")
    ply_data = np.empty(n_points, dtype=np.dtype([
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('opacity', '<f4')
    ]))
    ply_data['x'] = means_world[:, 0]
    ply_data['y'] = means_world[:, 1]
    ply_data['z'] = means_world[:, 2]
    ply_data['opacity'] = opacity_vals

    with open(ply_path, "wb") as f:
        header = (
            f"ply\n"
            f"format binary_little_endian 1.0\n"
            f"element vertex {n_points}\n"
            f"property float x\n"
            f"property float y\n"
            f"property float z\n"
            f"property float opacity\n"
            f"end_header\n"
        )
        f.write(header.encode("ascii"))
        f.write(ply_data.tobytes())

    print(f"Saved: {ply_path} ({n_points:,} points, with opacity [0,1])")


def do_export_poses(ckpt, output_dir, data_dir, test_every):
    """Export adjusted camera poses in COLMAP text format.

    Exports ALL images: training images get the learned pose adjustment applied,
    test images keep their original poses. IMAGE_IDs preserve the original COLMAP
    numbering.
    """
    from pycolmap import SceneManager

    parser = Parser(data_dir=data_dir, factor=1, normalize=False, test_every=test_every)

    n_images = len(parser.image_names)
    all_indices = np.arange(n_images)
    if test_every == 0:
        # test_every=0 means all images were used for training
        train_indices = all_indices
    else:
        train_indices = all_indices[all_indices % test_every != 0]
    n_train = len(train_indices)

    # Start with original camtoworlds for all images
    camtoworlds_all = parser.camtoworlds.copy()  # (n_images, 4, 4)

    # Apply pose adjustment to training images only
    if "pose_adjust" in ckpt:
        camtoworlds_train = torch.from_numpy(parser.camtoworlds[train_indices]).float()

        pose_adjust = CameraOptModule(n_train)
        pose_adjust.load_state_dict(ckpt["pose_adjust"])
        pose_adjust.eval()

        with torch.no_grad():
            embed_ids = torch.arange(n_train)
            adjusted = pose_adjust(camtoworlds_train, embed_ids)

        camtoworlds_all[train_indices] = adjusted.numpy()
        print(f"Applied pose adjustment to {n_train}/{n_images} training images")
    else:
        print("No pose_adjust in checkpoint, using original poses for all images")

    # Get original COLMAP image IDs (name -> colmap id)
    colmap_dir = os.path.join(data_dir, "sparse/0/")
    if not os.path.exists(colmap_dir):
        colmap_dir = os.path.join(data_dir, "sparse")

    manager = SceneManager(colmap_dir)
    manager.load_cameras()
    manager.load_images()

    imdata = manager.images
    name_to_colmap_id = {imdata[k].name: k for k in imdata}

    os.makedirs(output_dir, exist_ok=True)

    # --- Write images.txt ---
    images_txt_path = os.path.join(output_dir, "images.txt")
    with open(images_txt_path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {n_images}\n")

        for idx in range(n_images):
            c2w = camtoworlds_all[idx]
            w2c = np.linalg.inv(c2w)

            R_w2c = w2c[:3, :3]
            t_w2c = w2c[:3, 3]

            # Convert rotation matrix to quaternion (COLMAP convention: qw, qx, qy, qz)
            rot = Rotation.from_matrix(R_w2c)
            qxyzw = rot.as_quat()  # scipy returns [qx, qy, qz, qw]
            qvec = np.array([qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]])

            image_name = parser.image_names[idx]
            camera_id = parser.camera_ids[idx]
            image_id = name_to_colmap_id[image_name]

            f.write(
                f"{image_id} "
                f"{qvec[0]:.10f} {qvec[1]:.10f} {qvec[2]:.10f} {qvec[3]:.10f} "
                f"{t_w2c[0]:.10f} {t_w2c[1]:.10f} {t_w2c[2]:.10f} "
                f"{camera_id} {image_name}\n"
            )
            f.write("\n")  # empty POINTS2D line

    print(f"images.txt exported to {images_txt_path} "
          f"({n_images} images, {n_train} pose-adjusted)")

    # --- Copy or generate cameras.txt ---
    cameras_src = os.path.join(colmap_dir, "cameras.txt")
    cameras_dst = os.path.join(output_dir, "cameras.txt")

    if os.path.exists(cameras_src):
        shutil.copy2(cameras_src, cameras_dst)
        print(f"cameras.txt copied to {cameras_dst}")
    else:
        # Binary COLMAP format — regenerate cameras.txt from pycolmap
        _generate_cameras_txt(colmap_dir, cameras_dst)
        print(f"cameras.txt generated at {cameras_dst}")


# Map from pycolmap camera type ids to COLMAP model names and their param builders
_CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", lambda c: [c.fx, c.cx, c.cy]),
    1: ("PINHOLE", lambda c: [c.fx, c.fy, c.cx, c.cy]),
    2: ("SIMPLE_RADIAL", lambda c: [c.fx, c.cx, c.cy, c.k1]),
    3: ("RADIAL", lambda c: [c.fx, c.cx, c.cy, c.k1, c.k2]),
    4: ("OPENCV", lambda c: [c.fx, c.fy, c.cx, c.cy, c.k1, c.k2, c.p1, c.p2]),
    5: ("OPENCV_FISHEYE", lambda c: [c.fx, c.fy, c.cx, c.cy, c.k1, c.k2, c.k3, c.k4]),
}


def _generate_cameras_txt(colmap_dir, output_path):
    """Generate cameras.txt from binary COLMAP data via pycolmap."""
    from pycolmap import SceneManager

    manager = SceneManager(colmap_dir)
    manager.load_cameras()

    with open(output_path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(manager.cameras)}\n")

        for cam_id, cam in manager.cameras.items():
            cam_type = cam.camera_type
            if isinstance(cam_type, str):
                model_name = cam_type
                params_fn = None
                for _, (name, fn) in _CAMERA_MODELS.items():
                    if name == cam_type:
                        params_fn = fn
                        break
                if params_fn is None:
                    raise ValueError(f"Unsupported camera model: {cam_type}")
                params = params_fn(cam)
            else:
                if cam_type not in _CAMERA_MODELS:
                    raise ValueError(f"Unsupported camera type id: {cam_type}")
                model_name, params_fn = _CAMERA_MODELS[cam_type]
                params = params_fn(cam)

            params_str = " ".join(f"{p:.10f}" for p in params)
            f.write(f"{cam_id} {model_name} {cam.width} {cam.height} {params_str}\n")


def do_reset_step(ckpt, output_dir, ckpt_path):
    """Reset checkpoint step to 0 so it can be used with --ckpt for a fresh training run."""
    old_step = ckpt.get("step", "?")
    ckpt["step"] = 0

    if "pose_adjust" in ckpt:
        del ckpt["pose_adjust"]
        print("Removed pose_adjust from checkpoint")
    if "app_module" in ckpt:
        del ckpt["app_module"]
        print("Removed app_module from checkpoint")

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(ckpt_path)
    name_without_ext = os.path.splitext(base_name)[0]
    output_path = os.path.join(output_dir, f"{name_without_ext}_reset.pt")

    torch.save(ckpt, output_path)
    print(f"Checkpoint reset: step {old_step} -> 0")
    print(f"Saved to {output_path}")


def main():
    p = argparse.ArgumentParser(
        description="Export point clouds and COLMAP poses from a gsplat checkpoint"
    )
    p.add_argument("--ckpt", type=str, required=True, help="Path to .pt checkpoint file")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory (default: same directory as .pt file)")
    p.add_argument("--export_splat_ply", action="store_true", default=False,
                   help="Export standard 3DGS splat PLY (for Gaussian splatting viewers)")
    p.add_argument("--export_points3d", action="store_true", default=False,
                   help="Export COLMAP points3D.txt in metric coordinates")
    p.add_argument("--export_treeiso_points", action="store_true", default=False,
                   help="Export xyz+opacity PLY for treeiso input (sky removed, optional bbox crop)")
    p.add_argument("--bbox_ply", type=str, default=None,
                   help="PLY file whose XY minimum bounding rectangle is used to crop the export. "
                        "Z is ignored; the bounding box is not required to be axis-aligned.")
    p.add_argument("--bbox_margin", type=float, default=0.0,
                   help="Shrink each side of the bounding box inward by this many metres (default: 0.0)")
    p.add_argument("--export_poses", action="store_true", default=False,
                   help="Export camera poses in COLMAP text format (requires --data_dir)")
    p.add_argument("--reset_step", action="store_true", default=False,
                   help="Reset checkpoint step to 0 for re-training with --ckpt")
    p.add_argument("--data_dir", type=str, default=None,
                   help="COLMAP data directory (required for --export_poses)")
    p.add_argument("--test_every", type=int, default=8,
                   help="Train/test split parameter, must match training (default: 8)")
    args = p.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.splitext(os.path.abspath(args.ckpt))[0]

    if args.export_poses and args.data_dir is None:
        p.error("--data_dir is required when using --export_poses")

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"Checkpoint loaded (step {ckpt.get('step', '?')})")

    if args.export_splat_ply:
        do_export_splat_ply(ckpt, args.output_dir)

    if args.export_points3d:
        do_export_points3d(ckpt, args.output_dir)

    if args.export_treeiso_points:
        do_export_treeiso_points(ckpt, args.output_dir,
                                 bbox_ply=args.bbox_ply, bbox_margin=args.bbox_margin)

    if args.export_poses:
        do_export_poses(ckpt, args.output_dir, args.data_dir, args.test_every)

    if args.reset_step:
        do_reset_step(ckpt, args.output_dir, args.ckpt)

    print("Done.")


if __name__ == "__main__":
    main()
