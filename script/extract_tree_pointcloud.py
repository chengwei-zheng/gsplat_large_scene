#!/usr/bin/env python3
"""Extract a single tree's surface point cloud from a labeled gsplat checkpoint.

Pipeline:
  1. Load checkpoint + tree_labels (produced by assign_tree_labels.py).
  2. Extract Gaussians belonging to target tree label.
  3. Sample cameras on a fibonacci sphere around the tree AABB.
  4. Render per-camera depth maps (ED mode, only tree Gaussians).
  5. Back-project valid pixels to 3D, remove outliers, save .ply.

Usage:
    python script/extract_tree_pointcloud.py \\
        --ckpt  path/to/ckpt_labeled.pt \\
        --label 5 \\
        --output tree_5.ply
"""

import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from gsplat.rendering import rasterization


# ── Camera helpers ────────────────────────────────────────────────────────────

def fibonacci_sphere_with_elevation(n_views, elev_min_deg=-30.0, elev_max_deg=80.0):
    """Return (n_views, 3) unit vectors covering the given elevation band.

    Uses the fibonacci lattice so views are well-distributed on the sphere
    rather than clustered at poles.
    """
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    z_min = math.sin(math.radians(elev_min_deg))
    z_max = math.sin(math.radians(elev_max_deg))
    pts = []
    for i in range(n_views):
        z = z_min + (z_max - z_min) * (i + 0.5) / n_views
        r = math.sqrt(max(0.0, 1.0 - z * z))
        phi = 2.0 * math.pi * i / golden
        pts.append([r * math.cos(phi), r * math.sin(phi), z])
    return np.array(pts, dtype=np.float32)


def lookat_c2w(eye: np.ndarray, center: np.ndarray,
               world_up: np.ndarray = np.array([0.0, 0.0, 1.0])) -> np.ndarray:
    """Build a camera-to-world matrix in OpenCV convention (Y↓, Z forward).

    eye, center, world_up are in world coordinates.
    """
    z = center - eye
    z = z / (np.linalg.norm(z) + 1e-8)

    # Fall back to Y-up when camera is nearly vertical.
    if abs(float(np.dot(z, world_up))) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # OpenCV: X=right, Y=down, Z=forward
    # right = Z × world_up  (then Y = Z × X to stay right-handed)
    x = np.cross(z, world_up)
    x = x / (np.linalg.norm(x) + 1e-8)
    y = np.cross(z, x)
    y = y / (np.linalg.norm(y) + 1e-8)

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = x
    c2w[:3, 1] = y
    c2w[:3, 2] = z
    c2w[:3, 3] = eye
    return c2w


# ── Back-projection ───────────────────────────────────────────────────────────

def backproject(depth: np.ndarray, alpha: np.ndarray,
                K: np.ndarray, c2w: np.ndarray,
                alpha_thresh: float = 0.5,
                near: float = 1e-3,
                far: float = 1e4) -> np.ndarray:
    """Project valid depth pixels to world-space 3D points.

    depth : (H, W) – camera-space Z depth from gsplat ED mode
    alpha : (H, W) – alpha / accumulated opacity
    K     : (3, 3) – pinhole intrinsics
    c2w   : (4, 4) – camera-to-world (OpenCV)
    Returns (M, 3) array of world-space points.
    """
    H, W = depth.shape
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    # Pixel centre convention: same as gsplat (offset +0.5)
    xn = (xs + 0.5 - cx) / fx
    yn = (ys + 0.5 - cy) / fy

    valid = (alpha >= alpha_thresh) & (depth >= near) & (depth <= far)
    z_vals = depth[valid]            # (M,)
    xn_v   = xn[valid]
    yn_v   = yn[valid]

    # Back-project: ED depth = camera-space Z, so P_cam = [xn*z, yn*z, z]
    pts_cam = np.stack([xn_v * z_vals, yn_v * z_vals, z_vals], axis=-1)  # (M, 3)

    R = c2w[:3, :3]
    t = c2w[:3, 3]
    pts_world = pts_cam @ R.T + t    # (M, 3)
    return pts_world.astype(np.float32)


# ── Metric conversion ─────────────────────────────────────────────────────────

def denormalize_splats_inplace(splats: dict, transform) -> float:
    """Convert means / log-scales / quats from normalized to metric space.

    Mirrors simple_trainer._denormalize_splats but operates on raw (CPU)
    tensors stored in the splats dict, modifying them in-place.
    Returns the scale factor s so callers can log it.
    """
    if isinstance(transform, torch.Tensor):
        T = transform.float()
    else:
        T = torch.tensor(np.array(transform), dtype=torch.float32)

    s   = torch.linalg.norm(T[:3, 0])      # similarity scale
    R   = T[:3, :3] / s                     # pure rotation
    t   = T[:3, 3]                           # translation
    R_inv = R.T

    # means: P_metric = (P_norm - t) @ R_inv^T / s
    means = splats["means"].float()
    splats["means"] = (means - t[None, :]) @ R_inv.T / s

    # log-scales: exp(scale_metric) = exp(scale_norm) / s
    splats["scales"] = splats["scales"].float() - torch.log(s)

    # quats (wxyz): left-multiply by quaternion of R_inv
    trace = R_inv[0, 0] + R_inv[1, 1] + R_inv[2, 2]
    if trace > 0:
        rw = 0.5 * torch.sqrt(1.0 + trace)
        s4 = 0.25 / rw
        rx = (R_inv[2, 1] - R_inv[1, 2]) * s4
        ry = (R_inv[0, 2] - R_inv[2, 0]) * s4
        rz = (R_inv[1, 0] - R_inv[0, 1]) * s4
    else:
        rw = torch.tensor(1.0)
        rx = ry = rz = torch.tensor(0.0)

    quats = splats["quats"].float()
    qw, qx, qy, qz = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    splats["quats"] = torch.stack([
        rw*qw - rx*qx - ry*qy - rz*qz,
        rw*qx + rx*qw + ry*qz - rz*qy,
        rw*qy - rx*qz + ry*qw + rz*qx,
        rw*qz + rx*qy - ry*qx + rz*qw,
    ], dim=-1)

    return float(s)


# ── Statistical outlier removal ───────────────────────────────────────────────

def downsample_to_target(pts: np.ndarray, target: int,
                          method: str = "stride") -> np.ndarray:
    """Downsample pts to at most target points using the chosen method.

    method:
      "voxel"  – spatial voxel grid, binary-search for the right voxel size.
                 Preserves spatial distribution; slower.
      "stride" – keep every N-th point by index order.
                 Fast; distribution depends on how pts were concatenated.
      "random" – uniform random subset without replacement.
                 Fast; unbiased but non-deterministic.
    """
    if len(pts) <= target:
        return pts

    if method == "stride":
        step = max(1, len(pts) // target)
        return pts[::step]

    if method == "random":
        idx = np.random.choice(len(pts), size=target, replace=False)
        idx.sort()
        return pts[idx]

    # method == "voxel"
    aabb = pts.max(axis=0) - pts.min(axis=0)
    volume = float(aabb[0] * aabb[1] * aabb[2])
    voxel_size = (volume / target) ** (1.0 / 3.0) if volume > 0 else 1e-3

    def _downsample(v):
        coords = np.floor(pts / v).astype(np.int64)
        _, idx = np.unique(coords, axis=0, return_index=True)
        return pts[idx]

    lo, hi = 0.0, voxel_size * 4
    for _ in range(32):
        mid = (lo + hi) / 2
        if len(_downsample(mid)) <= target:
            hi = mid
        else:
            lo = mid
        if (hi - lo) / (hi + 1e-12) < 1e-3:
            break
    return _downsample(hi)


def remove_outliers_sor(pts: np.ndarray, k: int = 20, std_ratio: float = 2.0) -> np.ndarray:
    """Statistical outlier removal using a KD-tree (sklearn)."""
    if len(pts) <= k + 1:
        return pts
    from sklearn.neighbors import KDTree
    tree = KDTree(pts)
    dists, _ = tree.query(pts, k=k + 1)   # k+1 because first result is self
    mean_dist = dists[:, 1:].mean(axis=1)
    threshold = mean_dist.mean() + std_ratio * mean_dist.std()
    return pts[mean_dist <= threshold]


# ── Tree checkpoint saver ─────────────────────────────────────────────────────

def save_tree_ckpt(ckpt: dict, idx: torch.Tensor, label: int, path: str) -> None:
    """Save a single-tree checkpoint with the same structure as the source.

    All splat parameters are kept in their raw (unactivated) form so the file
    can be loaded directly by simple_trainer / simple_viewer.
    """
    splats = ckpt["splats"]
    tree_splats = {k: v[idx].clone() for k, v in splats.items()}

    tree_ckpt = {}
    # Copy non-splat metadata verbatim (transform, step, …)
    for k, v in ckpt.items():
        if k not in ("splats", "tree_labels"):
            tree_ckpt[k] = v
    tree_ckpt["splats"] = tree_splats
    # Mark every Gaussian with its label so downstream tools still work.
    tree_ckpt["tree_labels"] = torch.full((len(idx),), label, dtype=torch.int32)

    torch.save(tree_ckpt, path)
    print(f"Tree checkpoint saved → {path}  ({len(idx):,} Gaussians)")


# ── Depth image writer ────────────────────────────────────────────────────────

def save_depth_image(depth: np.ndarray, alpha: np.ndarray,
                     path: str, alpha_thresh: float = 0.5) -> None:
    """Save a false-colour depth image; invalid pixels are shown in black."""
    import cv2
    valid = alpha >= alpha_thresh
    vis = np.zeros(depth.shape, dtype=np.float32)
    if valid.any():
        d_valid = depth[valid]
        lo, hi = d_valid.min(), d_valid.max()
        if hi > lo:
            vis[valid] = (depth[valid] - lo) / (hi - lo)
        else:
            vis[valid] = 1.0
    # Convert to uint8 and apply a colour map for easy inspection.
    grey = (vis * 255).clip(0, 255).astype(np.uint8)
    colour = cv2.applyColorMap(grey, cv2.COLORMAP_TURBO)
    colour[~valid] = 0   # black background for invalid pixels
    cv2.imwrite(path, colour)


# ── PLY writer ────────────────────────────────────────────────────────────────

def save_ply(pts: np.ndarray, path: str) -> None:
    """Save Nx3 float32 point cloud as binary-little-endian PLY."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    pts = pts.astype(np.float32)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(pts)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(pts.tobytes())
    print(f"Saved {len(pts):,} points → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Extract a single tree's surface point cloud via depth rendering"
    )
    p.add_argument("--ckpt",    required=True, help="Labeled checkpoint (.pt with tree_labels)")
    p.add_argument("--label",   type=int, required=True, help="Tree label ID to extract")
    p.add_argument("--output",  default=None, help="Output .ply path")
    p.add_argument("--n_views", type=int,   default=64,   help="Number of camera views")
    p.add_argument("--dist_factor", type=float, default=2.0,
                   help="Camera distance = AABB-radius × dist_factor")
    p.add_argument("--resolution", type=int, default=1024, help="Render resolution (square)")
    p.add_argument("--fov",     type=float, default=60.0,  help="Vertical field of view (degrees)")
    p.add_argument("--alpha_thresh", type=float, default=0.2)
    p.add_argument("--near",    type=float, default=1e-3)
    p.add_argument("--far",     type=float, default=1e4)
    p.add_argument("--elev_min", type=float, default=-30.0, help="Min elevation (degrees)")
    p.add_argument("--elev_max", type=float, default=80.0,  help="Max elevation (degrees)")
    p.add_argument("--downsample_factor", type=float, default=20.0,
                   help="Downsample raw points to this multiple of Gaussian count (0=skip)")
    p.add_argument("--downsample_method", default="stride",
                   choices=["voxel", "stride", "random"],
                   help="Downsampling method: voxel / stride / random")
    p.add_argument("--sor_k",    type=int,   default=20,  help="SOR pass-1 neighbourhood size")
    p.add_argument("--sor_std",  type=float, default=1.5, help="SOR pass-1 std-dev multiplier")
    p.add_argument("--sor_k2",   type=int,   default=30,  help="SOR pass-2 neighbourhood size")
    p.add_argument("--sor_std2", type=float, default=1.5, help="SOR pass-2 std-dev multiplier")
    p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--metric", action="store_true",
                   help="Convert Gaussian parameters to metric coordinates using ckpt['transform']")
    p.add_argument("--debug", action="store_true",
                   help="Save intermediate results to <ckpt_dir>/<stem>_tree<label>_debug/")
    args = p.parse_args()

    ckpt_dir  = os.path.dirname(os.path.abspath(args.ckpt))
    ckpt_stem = os.path.splitext(os.path.basename(args.ckpt))[0]
    debug_dir = os.path.join(ckpt_dir, f"{ckpt_stem}_tree{args.label}_debug")

    if args.output is None:
        args.output = os.path.join(ckpt_dir, f"{ckpt_stem}_tree{args.label}.ply")

    if args.debug:
        os.makedirs(debug_dir, exist_ok=True)
        print(f"Debug dir: {debug_dir}")

    device = torch.device(args.device)

    # ── Step 1: Load checkpoint and extract tree Gaussians ────────────────────
    print("Loading checkpoint …")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    if "tree_labels" not in ckpt:
        raise KeyError("Checkpoint has no 'tree_labels'. Run assign_tree_labels.py first.")

    splats = ckpt["splats"]

    if args.metric:
        transform = ckpt.get("transform")
        if transform is None:
            print("Warning: no 'transform' in checkpoint, --metric has no effect.")
        else:
            s = denormalize_splats_inplace(splats, transform)
            print(f"Converted to metric coordinates (scale s={s:.6f})")
            # Set identity so the debug checkpoint is not transformed again downstream.
            ckpt["transform"] = torch.eye(4)

    labels = ckpt["tree_labels"].numpy()  # (N_total,)

    mask = labels == args.label
    n_tree = int(mask.sum())
    print(f"Tree label={args.label}: {n_tree:,} Gaussians")
    if n_tree == 0:
        available = np.unique(labels[labels >= 0]).tolist()
        raise ValueError(
            f"No Gaussians found for label {args.label}. "
            f"Available tree labels: {available[:20]}"
        )

    idx = torch.from_numpy(np.where(mask)[0])

    # ── [debug] Save single-tree checkpoint ───────────────────────────────────
    if args.debug:
        tree_ckpt_path = os.path.join(debug_dir, f"{ckpt_stem}_tree{args.label}.pt")
        save_tree_ckpt(ckpt, idx, args.label, tree_ckpt_path)

    means     = splats["means"][idx].to(device)                       # (M, 3)
    quats     = splats["quats"][idx].to(device)                       # (M, 4)
    scales    = torch.exp(splats["scales"][idx]).to(device)           # (M, 3)
    opacities = torch.sigmoid(splats["opacities"][idx]).to(device)    # (M,)

    if "sh0" in splats:
        colors = splats["sh0"][idx].to(device)                        # (M, 1, 3) DC only
    elif "colors" in splats:
        # Appearance-model checkpoint: use stored base colour as (M, 1, 3)
        c = torch.sigmoid(splats["colors"][idx])                      # (M, 3)
        colors = c.unsqueeze(1)
    else:
        colors = torch.zeros(n_tree, 1, 3, device=device)

    # ── Step 2: Compute AABB, build cameras ───────────────────────────────────
    means_np = means.detach().cpu().float().numpy()
    aabb_min  = means_np.min(axis=0)
    aabb_max  = means_np.max(axis=0)
    center    = ((aabb_min + aabb_max) / 2).astype(np.float32)
    radius    = float(np.linalg.norm(aabb_max - aabb_min) / 2)
    dist      = radius * args.dist_factor
    print(f"AABB center={center}, radius={radius:.4f}, camera dist={dist:.4f}")

    if args.debug:
        centers_ply = os.path.join(debug_dir, f"{ckpt_stem}_tree{args.label}_centers.ply")
        save_ply(means_np, centers_ply)

    W = H = args.resolution
    fy_val = H / (2.0 * math.tan(math.radians(args.fov / 2.0)))
    fx_val = fy_val
    K_np = np.array([[fx_val, 0, W / 2.0],
                     [0, fy_val, H / 2.0],
                     [0, 0, 1]], dtype=np.float32)
    K_t = torch.from_numpy(K_np).unsqueeze(0).to(device)             # (1, 3, 3)

    directions = fibonacci_sphere_with_elevation(
        args.n_views, args.elev_min, args.elev_max
    )

    # ── Step 3: Render depth maps and back-project ────────────────────────────
    all_pts = []
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    for i, d in enumerate(directions):
        eye = center + d.astype(np.float32) * dist
        c2w_np = lookat_c2w(eye, center, world_up)
        viewmat = torch.from_numpy(
            np.linalg.inv(c2w_np).astype(np.float32)
        ).unsqueeze(0).to(device)                                     # (1, 4, 4)

        with torch.no_grad():
            render_out, alphas, _ = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=viewmat,
                Ks=K_t,
                width=W,
                height=H,
                render_mode="ED",
                sh_degree=0,
                rasterize_mode="classic",
            )

        # render_out: (1, H, W, 1), alphas: (1, H, W, 1)
        depth_np = render_out[0, :, :, 0].cpu().numpy()
        alpha_np = alphas[0, :, :, 0].cpu().numpy()

        pts = backproject(depth_np, alpha_np, K_np, c2w_np,
                          alpha_thresh=args.alpha_thresh,
                          near=args.near, far=args.far)
        all_pts.append(pts)

        if args.debug:
            depth_img_path = os.path.join(debug_dir, f"depth_{i:04d}.png")
            save_depth_image(depth_np, alpha_np, depth_img_path, args.alpha_thresh)

        if (i + 1) % 16 == 0 or i == len(directions) - 1:
            total_so_far = sum(len(p) for p in all_pts)
            print(f"  View {i+1:3d}/{args.n_views}  pts this view={len(pts):,}  "
                  f"total={total_so_far:,}")

    if not all_pts or all(len(p) == 0 for p in all_pts):
        print("WARNING: no points collected across all views.")
        return

    all_pts = np.concatenate(all_pts, axis=0)
    print(f"Raw points: {len(all_pts):,}")

    if args.debug:
        raw_ply = os.path.join(debug_dir, f"{ckpt_stem}_tree{args.label}_raw.ply")
        save_ply(all_pts, raw_ply)

    # ── Step 4: Downsample then statistical outlier removal ──────────────────
    if args.downsample_factor > 0:
        target = int(n_tree * args.downsample_factor)
        all_pts = downsample_to_target(all_pts, target, method=args.downsample_method)
        print(f"After downsample/{args.downsample_method} (target={target:,}): {len(all_pts):,}")
        if args.debug:
            ds_ply = os.path.join(debug_dir, f"{ckpt_stem}_tree{args.label}_downsampled.ply")
            save_ply(all_pts, ds_ply)

    # ── SOR two passes ────────────────────────────────────────────────────────
    all_pts = remove_outliers_sor(all_pts, k=args.sor_k,  std_ratio=args.sor_std)
    print(f"After SOR-1 (k={args.sor_k}, std={args.sor_std}):  {len(all_pts):,}")
    all_pts = remove_outliers_sor(all_pts, k=args.sor_k2, std_ratio=args.sor_std2)
    print(f"After SOR-2 (k={args.sor_k2}, std={args.sor_std2}): {len(all_pts):,}")

    save_ply(all_pts, args.output)


if __name__ == "__main__":
    main()
