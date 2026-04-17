#!/usr/bin/env python3
"""Convert a standard 3DGS PLY file to a gsplat checkpoint (.pt file).

The output .pt file can be used with:
  - simple_viewer.py --ckpt output.pt           (interactive viewer)
  - simple_viewer.py --render_dataset ...       (batch render)
  - simple_trainer.py --resume output.pt        (continue training)

PLY field mapping (all values kept in their raw/log/logit space as-is):
  x, y, z          -> splats["means"]      (N, 3)
  f_dc_0..2        -> splats["sh0"]        (N, 1, 3)
  f_rest_0..3K-1   -> splats["shN"]        (N, K, 3)
  opacity          -> splats["opacities"]  (N,)   logit space
  scale_0..2       -> splats["scales"]     (N, 3)  log space
  rot_0..3         -> splats["quats"]      (N, 4)

If --data_dir is given, the same world-space normalization that simple_trainer.py
applies during data loading (Parser normalize=True) will be applied to the PLY,
so the output .pt is in the same coordinate system as checkpoints from that scene.
This is needed when the PLY and the COLMAP data share the same original coordinate
system (e.g. the PLY was reconstructed alongside the same point3D.txt).

Usage:
    # PLY already in normalized space (e.g. exported from this project's own .pt)
    python script/ply_to_ckpt.py --ply point_cloud.ply --output output.pt

    # PLY in original COLMAP space -> apply normalization to match training ckpts
    python script/ply_to_ckpt.py --ply point_cloud.ply --output output.pt \\
        --data_dir data/my_scene
"""

import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))


def load_ply(ply_path: str) -> dict:
    """Load a standard 3DGS binary PLY and return splats dict in checkpoint format."""
    with open(ply_path, "rb") as f:
        assert f.readline().strip() == b"ply", "Not a PLY file"
        fmt = f.readline().decode("ascii").strip()
        assert "binary_little_endian" in fmt, (
            f"Only binary_little_endian PLY is supported, got: {fmt}"
        )

        n_verts = None
        properties = []
        while True:
            line = f.readline().decode("ascii").strip()
            if line == "end_header":
                break
            if line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
            elif line.startswith("property float ") and n_verts is not None:
                properties.append(line.split()[-1])

        assert n_verts is not None, "Could not find 'element vertex' in PLY header"
        assert properties, "No float properties found under 'element vertex'"

        dtype = np.dtype([(p, np.float32) for p in properties])
        raw = np.frombuffer(f.read(n_verts * dtype.itemsize), dtype=dtype)

    print(f"Loaded {n_verts} gaussians, {len(properties)} properties")

    # means
    means = torch.from_numpy(
        np.stack([raw["x"], raw["y"], raw["z"]], axis=1).copy()
    )  # (N, 3)

    # sh0: f_dc_0, f_dc_1, f_dc_2  ->  (N, 1, 3)
    sh0 = torch.from_numpy(
        np.stack([raw["f_dc_0"], raw["f_dc_1"], raw["f_dc_2"]], axis=1).copy()
    ).unsqueeze(1)  # (N, 1, 3)

    # shN: f_rest_0 .. f_rest_{3K-1}
    # export direction: shN (N,K,3) -> permute(0,2,1) -> (N,3,K) -> reshape (N, 3K)
    # so f_rest_{0..3K-1} = [R_0..R_{K-1}, G_0..G_{K-1}, B_0..B_{K-1}]
    # reverse: (N, 3K) -> reshape (N, 3, K) -> permute(0, 2, 1) -> (N, K, 3)
    f_rest_props = sorted(
        [p for p in properties if p.startswith("f_rest_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    if f_rest_props:
        n_rest = len(f_rest_props)
        assert n_rest % 3 == 0, f"f_rest count {n_rest} is not divisible by 3"
        K = n_rest // 3
        f_rest = np.stack([raw[p] for p in f_rest_props], axis=1)  # (N, 3K)
        shN = (
            torch.from_numpy(f_rest.copy())
            .reshape(n_verts, 3, K)
            .permute(0, 2, 1)
            .contiguous()
        )  # (N, K, 3)
        sh_degree = int(math.sqrt(K + 1) - 1)
        print(f"SH degree: {sh_degree}  (K={K})")
    else:
        shN = torch.zeros(n_verts, 0, 3, dtype=torch.float32)
        print("No f_rest found, treating as SH degree 0")

    # scales: log space, stored as-is
    scales = torch.from_numpy(
        np.stack([raw["scale_0"], raw["scale_1"], raw["scale_2"]], axis=1).copy()
    )  # (N, 3)

    # quats: not normalized, stored as-is
    quats = torch.from_numpy(
        np.stack([raw["rot_0"], raw["rot_1"], raw["rot_2"], raw["rot_3"]], axis=1).copy()
    )  # (N, 4)

    # opacities: logit space, stored as-is
    opacities = torch.from_numpy(raw["opacity"].copy())  # (N,)

    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "sh0": sh0,
        "shN": shN,
    }


def apply_transform(splats: dict, T: np.ndarray) -> dict:
    """Apply a 4x4 similarity transform T (from Parser.transform) to the splats.

    T encodes:  p_normalized = s * R @ p_original + t
    where T[:3, :3] = s * R  and  T[:3, 3] = t.

    Effects on each Gaussian parameter:
      means:  direct point transform
      scales: uniform scale s multiplies all axes -> add log(s) in log space
      quats:  pure rotation R rotates the Gaussian orientation -> R_quat ⊗ q
      sh0/shN/opacities: unaffected (appearance, not geometry)
    """
    from scipy.spatial.transform import Rotation

    sR = T[:3, :3]  # s * R
    t = T[:3, 3]
    s = float(np.linalg.norm(sR[0]))  # scale factor (norm of any row)
    R = sR / s  # pure rotation matrix

    # means: p_new = s * R @ p + t  (same as transform_points)
    means_np = splats["means"].numpy().astype(np.float64)
    means_new = means_np @ sR.T + t
    splats["means"] = torch.from_numpy(means_new.astype(np.float32))

    # scales: log(scale_new) = log(s * scale_old) = log(scale_old) + log(s)
    splats["scales"] = splats["scales"] + float(np.log(s))

    # quats: compose R with existing quaternion  q_new = r_quat ⊗ q
    # gsplat uses [w, x, y, z] convention (rot_0 = w)
    r_xyzw = Rotation.from_matrix(R).as_quat()  # scipy: [x, y, z, w]
    r_wxyz = np.array([r_xyzw[3], r_xyzw[0], r_xyzw[1], r_xyzw[2]], dtype=np.float32)
    r_tensor = torch.from_numpy(r_wxyz)  # (4,)

    # Quaternion multiplication: (r_wxyz) ⊗ (q_wxyz)
    # [w1,x1,y1,z1] ⊗ [w2,x2,y2,z2]
    q = splats["quats"]  # (N, 4), [w, x, y, z]
    rw, rx, ry, rz = r_tensor[0], r_tensor[1], r_tensor[2], r_tensor[3]
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    new_w = rw * qw - rx * qx - ry * qy - rz * qz
    new_x = rw * qx + rx * qw + ry * qz - rz * qy
    new_y = rw * qy - rx * qz + ry * qw + rz * qx
    new_z = rw * qz + rx * qy - ry * qx + rz * qw
    splats["quats"] = torch.stack([new_w, new_x, new_y, new_z], dim=1)

    print(f"Applied transform: scale={s:.6f}, t={t}")
    return splats


def main():
    p = argparse.ArgumentParser(
        description="Convert a standard 3DGS PLY file to a gsplat checkpoint (.pt)"
    )
    p.add_argument("--ply", type=str, required=True, nargs="+", help="Input PLY file path(s). Provide two paths to merge them.")
    p.add_argument("--output", type=str, required=True, help="Output .pt file path")
    p.add_argument(
        "--step",
        type=int,
        default=0,
        help="Step stored in checkpoint (default: 0). "
             "Use a non-zero value with --resume to start schedulers at the right point.",
    )
    p.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="COLMAP data directory. If given, applies the same world normalization "
             "as simple_trainer.py (Parser normalize=True) so the output .pt matches "
             "the coordinate system of checkpoints trained on this scene.",
    )
    p.add_argument(
        "--test_every",
        type=int,
        default=8,
        help="test_every used during training, needed to load Parser (default: 8)",
    )
    args = p.parse_args()

    if len(args.ply) > 2:
        p.error("At most two PLY files can be provided.")

    splats = load_ply(args.ply[0])
    if len(args.ply) == 2:
        print(f"\nMerging with second PLY: {args.ply[1]}")
        splats2 = load_ply(args.ply[1])
        splats = {k: torch.cat([splats[k], splats2[k]], dim=0) for k in splats}
        print(f"Merged total: {splats['means'].shape[0]} gaussians")

    if args.data_dir is not None:
        from datasets.colmap import Parser
        print(f"Loading Parser from {args.data_dir} to get normalization transform...")
        parser = Parser(
            data_dir=args.data_dir,
            factor=1,
            normalize=True,
            test_every=args.test_every,
        )
        T = parser.transform  # (4, 4) similarity transform
        print(f"Transform matrix:\n{T}")
        splats = apply_transform(splats, T)

    ckpt = {"step": args.step, "splats": splats}
    torch.save(ckpt, args.output)
    print(f"Saved to {args.output}  (step={args.step})")


if __name__ == "__main__":
    main()
