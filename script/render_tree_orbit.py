#!/usr/bin/env python3
"""Render a 360° orbit video around a single-tree gsplat checkpoint.

The camera flies at a fixed height above the tree's lowest Gaussian and at a
fixed horizontal distance from the tree's XY centre, rotating a full circle.

Usage:
    python script/render_tree_orbit.py \\
        --ckpt  path/to/ckpt_labeled_tree5.pt \\
        --height 1.5 \\
        --dist   3.0
"""

import argparse
import math
import os
import sys

import imageio
import json
import numpy as np
import torch
import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from gsplat.rendering import rasterization


# ── Camera helper ─────────────────────────────────────────────────────────────

def lookat_c2w(eye: np.ndarray, center: np.ndarray,
               world_up: np.ndarray = np.array([0.0, 0.0, 1.0])) -> np.ndarray:
    """OpenCV c2w: X right, Y down, Z forward."""
    z = center - eye
    z = z / (np.linalg.norm(z) + 1e-8)
    if abs(float(np.dot(z, world_up))) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
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


# ── Depth visualisation ───────────────────────────────────────────────────────

def depth_to_colormap(depth: np.ndarray, alpha: np.ndarray,
                      alpha_thresh: float = 0.5) -> np.ndarray:
    """Return a uint8 (H, W, 3) RGB false-colour depth image (TURBO colormap).

    Valid pixels are normalised to [0, 1] within the frame; invalid pixels
    (alpha < alpha_thresh) are shown in black.
    """
    import cv2
    valid = alpha >= alpha_thresh
    vis   = np.zeros(depth.shape, dtype=np.float32)
    if valid.any():
        lo, hi = depth[valid].min(), depth[valid].max()
        vis[valid] = (depth[valid] - lo) / (hi - lo + 1e-8)
    grey   = (vis * 255).clip(0, 255).astype(np.uint8)
    colour = cv2.applyColorMap(grey, cv2.COLORMAP_PLASMA)  # BGR
    colour = cv2.cvtColor(colour, cv2.COLOR_BGR2RGB)        # → RGB
    colour[~valid] = 0
    return colour


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Render a 360° orbit video around a single-tree gsplat checkpoint"
    )
    p.add_argument("--ckpt",       required=True, help="Single-tree checkpoint (.pt)")
    p.add_argument("--height",     type=float, default=1.5,
                   help="Camera height above the tree's lowest Gaussian (z_min + height)")
    p.add_argument("--dist",       type=float, default=10.0,
                   help="Horizontal distance from the tree's XY centre")
    p.add_argument("--n_frames",   type=int,   default=120,
                   help="Number of frames for a full 360° orbit")
    p.add_argument("--resolution", type=int,   nargs="+", default=[1280],
                   metavar="RES",
                   help="Render resolution: one value for a square frame (e.g. 1280) "
                        "or two for width height (e.g. 1280 720)")
    p.add_argument("--fov",        type=float, default=60.0,
                   help="Vertical field of view (degrees)")
    p.add_argument("--fps",        type=int,   default=30)
    p.add_argument("--lookat_height", type=float, default=2.0,
                   help="Height of look-at target above z_floor (default: 2.0)")
    p.add_argument("--sh_degree",  type=int,   default=None,
                   help="SH degree for colour rendering (default: inferred from checkpoint)")
    p.add_argument("--output_dir", default=None,
                   help="Output directory (default: same folder and stem as --ckpt)")
    p.add_argument("--save_cameras", action="store_true",
                   help="Save per-frame c2w and K to cameras.json in output_dir")
    p.add_argument("--depth", action="store_true",
                   help="Render depth map alongside RGB (side by side)")
    p.add_argument("--min_opacity", type=float, default=0.0,
                   help="Remove Gaussians with opacity below this threshold (default: 0, keep all)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if len(args.resolution) == 1:
        args.width, args.height_px = args.resolution[0], args.resolution[0]
    elif len(args.resolution) == 2:
        args.width, args.height_px = args.resolution
    else:
        p.error("--resolution takes either 1 value (square) or 2 values (width height)")

    # Default output dir: same directory, same stem as the input .pt
    if args.output_dir is None:
        ckpt_abs  = os.path.abspath(args.ckpt)
        stem      = os.path.splitext(ckpt_abs)[0]
        args.output_dir = stem + "_render"   # e.g. /path/to/ckpt_labeled_tree5_render

    os.makedirs(args.output_dir, exist_ok=True)
    frames_dir = os.path.join(args.output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    device = torch.device(args.device)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print("Loading checkpoint …")
    ckpt   = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    splats = ckpt["splats"]

    means     = splats["means"].to(device)                        # (N, 3)
    quats     = splats["quats"].to(device)                        # (N, 4)
    scales    = torch.exp(splats["scales"]).to(device)            # (N, 3)
    opacities = torch.sigmoid(splats["opacities"]).to(device)     # (N,)

    if "sh0" in splats:
        sh0 = splats["sh0"].to(device)                            # (N, 1, 3)
        shN = splats["shN"].to(device)                            # (N, K, 3)
        colors = torch.cat([sh0, shN], dim=1)                     # (N, 1+K, 3)
        if args.sh_degree is None:
            total_coeffs = colors.shape[1]                        # (sh_degree+1)^2
            args.sh_degree = int(math.isqrt(total_coeffs)) - 1
    elif "colors" in splats:
        print("Warning: appearance-model checkpoint — using base colour only (no view-dependent shading).")
        c      = torch.sigmoid(splats["colors"]).to(device)       # (N, 3)
        colors = c.unsqueeze(1)                                    # (N, 1, 3)
        args.sh_degree = 0
    else:
        colors = torch.zeros(len(means), 1, 3, device=device)
        args.sh_degree = 0

    if args.min_opacity > 0.0:
        keep = opacities >= args.min_opacity
        means, quats, scales, opacities, colors = (
            means[keep], quats[keep], scales[keep], opacities[keep], colors[keep]
        )
        print(f"Opacity filter >= {args.min_opacity}: {keep.sum():,} / {keep.shape[0]:,} kept")

    print(f"Gaussians: {len(means):,}  sh_degree: {args.sh_degree}")

    # ── Tree geometry ─────────────────────────────────────────────────────────
    means_np = means.detach().cpu().float().numpy()
    z_floor  = float(means_np[:, 2].min())
    aabb_min = means_np.min(axis=0)
    aabb_max = means_np.max(axis=0)
    aabb_center = (aabb_min + aabb_max) / 2
    center = np.array([aabb_center[0], aabb_center[1], z_floor + args.lookat_height],
                      dtype=np.float32)

    cam_z  = z_floor + args.height
    print(f"z_floor={z_floor:.4f}  camera z={cam_z:.4f}  dist={args.dist}")
    print(f"Look-at target: {center}")

    # ── Intrinsics ────────────────────────────────────────────────────────────
    W, H    = args.width, args.height_px
    fy_val  = H / (2.0 * math.tan(math.radians(args.fov / 2.0)))
    K_np    = np.array([[fy_val, 0, W / 2.0],
                        [0, fy_val, H / 2.0],
                        [0,      0,      1.0]], dtype=np.float32)
    K_t     = torch.from_numpy(K_np).unsqueeze(0).to(device)     # (1, 3, 3)

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # ── Render loop ───────────────────────────────────────────────────────────
    video_path = os.path.join(args.output_dir, "orbit.mp4")
    writer     = imageio.get_writer(video_path, fps=args.fps)
    cameras    = []   # collected when --save_cameras

    for i in tqdm.trange(args.n_frames, desc="Rendering"):
        theta = 2.0 * math.pi * i / args.n_frames
        eye   = np.array([
            center[0] + args.dist * math.cos(theta),
            center[1] + args.dist * math.sin(theta),
            cam_z,
        ], dtype=np.float32)

        c2w = lookat_c2w(eye, center, world_up)

        if args.save_cameras:
            cameras.append({
                "frame": i,
                "c2w": c2w.tolist(),
                "K":   K_np.tolist(),
                "width":  W,
                "height": H,
            })

        viewmat = torch.from_numpy(
            np.linalg.inv(c2w).astype(np.float32)
        ).unsqueeze(0).to(device)                                 # (1, 4, 4)

        render_mode = "RGB+ED" if args.depth else "RGB"
        with torch.no_grad():
            render_out, render_alphas, _ = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=viewmat,
                Ks=K_t,
                width=W,
                height=H,
                render_mode=render_mode,
                sh_degree=args.sh_degree,
                rasterize_mode="classic",
            )

        rgb    = render_out[0, :, :, :3].clamp(0.0, 1.0).cpu().numpy()  # (H, W, 3)
        rgb_u8 = (rgb * 255).astype(np.uint8)

        if args.depth:
            depth    = render_out[0, :, :, 3].cpu().numpy()              # (H, W)
            alpha    = render_alphas[0, :, :, 0].cpu().numpy()           # (H, W)
            depth_u8 = depth_to_colormap(depth, alpha)
            canvas   = np.concatenate([rgb_u8, depth_u8], axis=1)       # (H, 2W, 3)
        else:
            canvas = rgb_u8

        frame_path = os.path.join(frames_dir, f"{i:04d}.png")
        imageio.imwrite(frame_path, canvas)
        writer.append_data(canvas)

    writer.close()

    if args.save_cameras:
        cam_path = os.path.join(args.output_dir, "cameras.json")
        with open(cam_path, "w") as f:
            json.dump(cameras, f, indent=2)
        print(f"Cameras → {cam_path}")

    print(f"Frames → {frames_dir}/")
    print(f"Video  → {video_path}")


if __name__ == "__main__":
    main()
