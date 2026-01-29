#!/usr/bin/env python3
"""Export PLY and COLMAP poses from a gsplat checkpoint (.pt file).

Usage:
    # Export PLY only
    python script/export_ckpt.py --ckpt results/xxx/ckpts/ckpt_29999_rank0.pt

    # Export PLY + poses
    python script/export_ckpt.py --ckpt results/xxx/ckpts/ckpt_29999_rank0.pt --data_dir data/xxx

    # Disable PLY export, only export poses
    python script/export_ckpt.py --ckpt results/xxx/ckpts/ckpt_29999_rank0.pt --data_dir data/xxx --no_export_ply
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


def do_export_ply(ckpt, output_dir):
    """Export splats from checkpoint as a standard 3DGS PLY file.

    Scales and opacities are kept in log/logit space (matching original 3DGS convention).
    Quaternions are not normalized.
    """
    splats = ckpt["splats"]

    means = splats["means"]
    scales = splats["scales"]
    quats = splats["quats"]
    opacities = splats["opacities"]

    if "sh0" not in splats:
        print("Warning: checkpoint uses appearance model (no sh0/shN). Skipping PLY export.")
        return

    sh0 = splats["sh0"]
    shN = splats["shN"]

    os.makedirs(output_dir, exist_ok=True)
    ply_path = os.path.join(output_dir, "point_cloud.ply")

    export_splats(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        shN=shN,
        format="ply",
        save_to=ply_path,
    )
    print(f"PLY exported to {ply_path} ({len(means)} gaussians)")


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
                # Find the param builder by name
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


def main():
    p = argparse.ArgumentParser(
        description="Export PLY and COLMAP poses from a gsplat checkpoint"
    )
    p.add_argument("--ckpt", type=str, required=True, help="Path to .pt checkpoint file")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory (default: same directory as .pt file)")
    p.add_argument("--export_ply", action=argparse.BooleanOptionalAction, default=True,
                   help="Export PLY (default: True)")
    p.add_argument("--export_poses", action=argparse.BooleanOptionalAction, default=True,
                   help="Export poses (default: True, requires --data_dir)")
    p.add_argument("--data_dir", type=str, default=None,
                   help="COLMAP data directory (required for pose export)")
    p.add_argument("--test_every", type=int, default=8,
                   help="Train/test split parameter, must match training (default: 8)")
    args = p.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(args.ckpt))

    if args.export_poses and args.data_dir is None:
        p.error("--data_dir is required when exporting poses")

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    print(f"Checkpoint loaded (step {ckpt.get('step', '?')})")

    if args.export_ply:
        do_export_ply(ckpt, args.output_dir)

    if args.export_poses:
        do_export_poses(ckpt, args.output_dir, args.data_dir, args.test_every)

    print("Done.")


if __name__ == "__main__":
    main()
