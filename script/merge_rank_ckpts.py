"""
Merge multi-rank checkpoints from a single multi-GPU training run into one .pt file.

During multi-GPU training, each rank saves its own checkpoint:
    ckpts/ckpt_{step}_rank{r}.pt

This script concatenates the splat tensors from all ranks and writes a single
merged checkpoint that can be used for single-GPU evaluation / export.

Usage:
    python merge_rank_ckpts.py --ckpt_dir /path/to/result/ckpts [--step 29999] [--output merged.pt]
    python merge_rank_ckpts.py --ckpt_dir /path/to/result/ckpts --sparse_dir /path/to/sparse [--remove_sky]

    # Patch an existing merged checkpoint that lacks 'transform' (old format):
    python merge_rank_ckpts.py --ckpt path/to/merged.pt --sparse_dir /path/to/sparse [--remove_sky]
"""

import argparse
import os
import re

import numpy as np
import torch


def _resolve_sparse_dir(sparse_dir):
    """Return the directory that directly contains points3D.bin/.txt."""
    candidate = sparse_dir
    if not (
        os.path.exists(os.path.join(candidate, "points3D.bin"))
        or os.path.exists(os.path.join(candidate, "points3D.txt"))
    ):
        for sub in sorted(os.listdir(candidate)):
            sub_path = os.path.join(candidate, sub)
            if os.path.isdir(sub_path) and (
                os.path.exists(os.path.join(sub_path, "points3D.bin"))
                or os.path.exists(os.path.join(sub_path, "points3D.txt"))
            ):
                candidate = sub_path
                break
        else:
            raise FileNotFoundError(
                f"Could not find points3D.bin/.txt under {sparse_dir}"
            )
    return candidate


def read_colmap_points(sparse_dir):
    """Return (N, 3) float32 array of points3D XYZ from a COLMAP sparse folder."""
    from pycolmap import SceneManager
    candidate = _resolve_sparse_dir(sparse_dir)
    print(f"Reading COLMAP sparse from: {candidate}")
    sm = SceneManager(candidate)
    sm.load_points3D()
    return sm.points3D.astype(np.float32)


def infer_transform_from_sparse(sparse_dir):
    """Recompute the (4,4) normalization transform from COLMAP cameras + points.

    Reproduces the exact T1 → T2 → T3 pipeline used by datasets/colmap.py Parser
    so the result is consistent with what simple_trainer stores in checkpoints.
    Assumes normalize_world_space=True (the training default).
    """
    from pycolmap import SceneManager
    from datasets.normalize import (
        similarity_from_cameras,
        align_principal_axes,
        transform_points,
        transform_cameras,
    )

    candidate = _resolve_sparse_dir(sparse_dir)
    sm = SceneManager(candidate)
    sm.load_cameras()
    sm.load_images()
    sm.load_points3D()

    # Build camtoworld matrices (same as Parser)
    bottom = np.array([0, 0, 0, 1], dtype=np.float64).reshape(1, 4)
    w2c_mats = []
    for im in sm.images.values():
        rot = im.R()
        trans = im.tvec.reshape(3, 1)
        w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
        w2c_mats.append(w2c)
    camtoworlds = np.linalg.inv(np.stack(w2c_mats, axis=0))  # (N, 4, 4)

    points = sm.points3D.astype(np.float32)  # (M, 3)

    # T1: similarity from cameras
    T1 = similarity_from_cameras(camtoworlds)
    camtoworlds = transform_cameras(T1, camtoworlds)
    points = transform_points(T1, points)

    # T2: align principal axes
    T2 = align_principal_axes(points)
    points = transform_points(T2, points)
    transform = T2 @ T1

    # T3: flip if upside-down
    if np.median(points[:, 2]) > np.mean(points[:, 2]):
        T3 = np.array([
            [1.0,  0.0,  0.0, 0.0],
            [0.0, -1.0,  0.0, 0.0],
            [0.0,  0.0, -1.0, 0.0],
            [0.0,  0.0,  0.0, 1.0],
        ])
        transform = T3 @ transform

    return torch.from_numpy(transform.astype(np.float32))


def detect_sky_gaussians(means, transform, colmap_pts):
    """Return (sky_mask, sky_hemisphere_center, sky_depth_min).

    sky_mask              : boolean tensor (N,), True = sky Gaussian
    sky_hemisphere_center : float32 tensor (3,) in normalized space
    sky_depth_min         : float, distance threshold in normalized space

    Uses the same hemisphere logic as simple_trainer:
      sky_hemisphere_center = [x_center, y_center, z_min]  (normalized space)
      sky_depth_min         = 0.5 * max_z                  (normalized space)
      sky mask              = dist(mean, center) > sky_depth_min
    """
    # Apply transform to COLMAP points → normalized space
    # transform: (4, 4),  p_norm = p_colmap @ R.T + t
    T = transform.numpy() if isinstance(transform, torch.Tensor) else transform
    pts_norm = colmap_pts @ T[:3, :3].T + T[:3, 3]  # (N, 3)

    max_z = float(pts_norm[:, 2].max())
    z_min = float(pts_norm[:, 2].min())
    x_center = float((pts_norm[:, 0].min() + pts_norm[:, 0].max()) / 2)
    y_center = float((pts_norm[:, 1].min() + pts_norm[:, 1].max()) / 2)
    sky_depth_min = 0.5 * max_z
    center = torch.tensor([x_center, y_center, z_min], dtype=means.dtype)

    print(f"Point cloud Z range (normalized): [{z_min:.4f}, {max_z:.4f}]")
    print(f"Sky hemisphere center: [{x_center:.4f}, {y_center:.4f}, {z_min:.4f}], radius threshold: {sky_depth_min:.4f}")

    dist = torch.norm(means - center[None, :], dim=-1)  # (N,)
    sky_mask = dist > sky_depth_min
    return sky_mask, center, sky_depth_min


def find_steps(ckpt_dir):
    """Return sorted list of steps that have at least a rank0 checkpoint."""
    steps = set()
    for fname in os.listdir(ckpt_dir):
        m = re.match(r"ckpt_(\d+)_rank\d+\.pt$", fname)
        if m:
            steps.add(int(m.group(1)))
    return sorted(steps)


def find_ranks(ckpt_dir, step):
    """Return sorted list of rank indices present for a given step."""
    ranks = []
    for fname in os.listdir(ckpt_dir):
        m = re.match(rf"ckpt_{step}_rank(\d+)\.pt$", fname)
        if m:
            ranks.append(int(m.group(1)))
    return sorted(ranks)


def main():
    parser = argparse.ArgumentParser(description="Merge multi-rank .pt checkpoints into one")
    # --- source: either rank files (--ckpt_dir) or an already-merged single file (--ckpt) ---
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt_dir",
                     help="Directory containing ckpt_*_rank*.pt files")
    src.add_argument("--ckpt",
                     help="Path to an already-merged .pt file (skips rank merging; "
                          "useful for patching old checkpoints that lack 'transform')")
    parser.add_argument("--step", type=int, default=None,
                        help="Training step to merge (default: latest); ignored with --ckpt")
    parser.add_argument("--output", default=None,
                        help="Output file path (default: <ckpt_dir>/ckpt_<step>_merged.pt "
                             "or <ckpt_stem>_patched.pt when using --ckpt)")
    parser.add_argument("--device", default="cpu",
                        help="Device for tensor ops (default: cpu)")
    parser.add_argument("--clamp_scale", type=float, default=None,
                        help="Clamp Gaussian scale (σ) to this maximum value after merging")
    parser.add_argument("--sparse_dir", default=None,
                        help="Path to COLMAP sparse folder (e.g. .../sparse/0 or .../sparse). "
                             "Used to read the point cloud and detect sky Gaussians with the "
                             "same hemisphere method as simple_trainer.")
    parser.add_argument("--remove_sky", action="store_true",
                        help="Remove detected sky Gaussians from the merged checkpoint")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ------------------------------------------------------------------
    # Build merged checkpoint
    # ------------------------------------------------------------------
    if args.ckpt is not None:
        # --- Mode: patch an already-merged single checkpoint ---
        if not os.path.isfile(args.ckpt):
            raise FileNotFoundError(f"ckpt not found: {args.ckpt}")
        print(f"Loading existing merged checkpoint: {args.ckpt}")
        merged = torch.load(args.ckpt, map_location=device, weights_only=False)
        merged_splats = merged["splats"]
        step = merged.get("step", "?")
        print(f"Checkpoint step: {step}")
        default_out = os.path.splitext(os.path.abspath(args.ckpt))[0] + "_patched.pt"
    else:
        # --- Mode: merge from per-rank files ---
        ckpt_dir = args.ckpt_dir
        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f"ckpt_dir not found: {ckpt_dir}")

        steps = find_steps(ckpt_dir)
        if not steps:
            raise RuntimeError(f"No ckpt_*_rank*.pt files found in {ckpt_dir}")

        step = args.step if args.step is not None else steps[-1]
        if step not in steps:
            raise ValueError(f"Step {step} not found. Available steps: {steps}")

        ranks = find_ranks(ckpt_dir, step)
        if not ranks:
            raise RuntimeError(f"No rank files found for step {step}")

        print(f"Step:  {step}")
        print(f"Ranks: {ranks}")

        rank_ckpts = []
        for r in ranks:
            path = os.path.join(ckpt_dir, f"ckpt_{step}_rank{r}.pt")
            print(f"  Loading rank {r}: {path}")
            ck = torch.load(path, map_location=device, weights_only=False)
            rank_ckpts.append(ck)

        splat_keys = list(rank_ckpts[0]["splats"].keys())
        print(f"Splat keys: {splat_keys}")

        merged_splats = {}
        for k in splat_keys:
            tensors = [ck["splats"][k] for ck in rank_ckpts]
            merged_splats[k] = torch.cat(tensors, dim=0)
            print(f"  {k}: {tensors[0].shape} x {len(ranks)} -> {merged_splats[k].shape}")

        transform_src = rank_ckpts[0].get("transform", None)
        if transform_src is None:
            print("Warning: checkpoint has no 'transform' key (old format).")
        merged = {"step": step, "splats": merged_splats}
        if transform_src is not None:
            merged["transform"] = transform_src
        for optional_key in ("pose_adjust", "app_module", "bil_grids", "bilateral_grid_shape",
                              "sky_hemisphere_center", "sky_depth_min"):
            if optional_key in rank_ckpts[0]:
                merged[optional_key] = rank_ckpts[0][optional_key]

        default_out = os.path.join(ckpt_dir, f"ckpt_{step}_merged.pt")

    total_gs = merged_splats["means"].shape[0]
    print(f"Total Gaussians: {total_gs:,}")

    # ------------------------------------------------------------------
    # Infer / propagate transform, then detect / remove sky
    # ------------------------------------------------------------------
    transform = merged.get("transform", None)

    if args.sparse_dir is not None:
        if transform is None:
            print("Warning: checkpoint has no 'transform'; inferring normalization transform from sparse cameras+points.")
            transform = infer_transform_from_sparse(args.sparse_dir)
            print(f"Inferred transform:\n{transform.numpy()}")
            merged["transform"] = transform  # write inferred transform into the output checkpoint

        if "sky_hemisphere_center" in merged and "sky_depth_min" in merged:
            print("Sky parameters already in checkpoint; skipping recomputation.")
            sky_center = merged["sky_hemisphere_center"]
            sky_depth_min = float(merged["sky_depth_min"])
            dist = torch.norm(merged_splats["means"] - sky_center[None, :], dim=-1)
            sky_mask = dist > sky_depth_min
        else:
            colmap_pts = read_colmap_points(args.sparse_dir)
            sky_mask, sky_center, sky_depth_min = detect_sky_gaussians(merged_splats["means"], transform, colmap_pts)
            merged["sky_hemisphere_center"] = sky_center
            merged["sky_depth_min"] = torch.tensor(sky_depth_min)
        n_sky = sky_mask.sum().item()
        print(f"Sky Gaussians detected: {n_sky:,} / {total_gs:,} ({100.0 * n_sky / total_gs:.2f}%)")
        if args.remove_sky:
            keep = ~sky_mask
            for k in merged_splats:
                merged_splats[k] = merged_splats[k][keep]
            merged["splats"] = merged_splats
            n_kept = keep.sum().item()
            print(f"Removed {n_sky:,} sky Gaussians. Remaining: {n_kept:,}")
    elif args.remove_sky:
        print("Warning: --remove_sky has no effect without --sparse_dir")

    if args.clamp_scale is not None:
        total_gs = merged_splats["means"].shape[0]
        log_max = torch.tensor(args.clamp_scale, dtype=merged_splats["scales"].dtype).log()
        n_clamped = (merged_splats["scales"] > log_max).any(dim=-1).sum().item()
        merged_splats["scales"] = merged_splats["scales"].clamp(max=log_max)
        print(f"[clamp_scale={args.clamp_scale}] Clamped {n_clamped:,} Gaussians ({100*n_clamped/total_gs:.2f}%)")

    out_path = args.output or default_out
    torch.save(merged, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
