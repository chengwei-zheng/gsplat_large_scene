"""
Project COLMAP points3D onto camera views and save rendered images.

Usage:
    python project_pointcloud.py \
        --sparse_dir /path/to/sparse/0 \
        --output_dir /path/to/output \
        [--camera_id 0]       # only process images from this camera (default: all)
        [--step 10]           # process every Nth image (default: 1)
        [--point_size 2]      # point radius in pixels, used when --sphere is off (default: 1)
        [--bg_color white]    # background color: black or white (default: black)
        [--sphere]            # render points as depth-scaled spheres (closer = larger)
        [--sphere_scale 5.0]  # radius = sphere_scale / depth (default: 5.0)
        [--max_radius 20]     # maximum sphere radius in pixels (default: 20)
        [--gpu]               # use GPU (PyTorch CUDA) for rendering
        [--ground_white]      # paint background pixels whose ray points downward as white
                              # assumes Z-up world coordinates
"""

import argparse
import os
import numpy as np
import cv2
import pycolmap
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparse_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--camera_id", type=int, default=None)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--point_size", type=int, default=1, help="Fixed point radius (used when --sphere is off)")
    parser.add_argument("--bg_color", choices=["black", "white"], default="black")
    parser.add_argument("--sphere", action="store_true", help="Render depth-scaled spheres")
    parser.add_argument("--sphere_scale", type=float, default=5.0, help="radius = sphere_scale / depth")
    parser.add_argument("--max_radius", type=int, default=20, help="Max sphere radius in pixels")
    parser.add_argument("--gpu", action="store_true", help="Use GPU (CUDA) for rendering")
    parser.add_argument("--ground_white", action="store_true",
                        help="Paint downward-looking background pixels white (assumes Z-up world coords)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# CPU rendering
# ---------------------------------------------------------------------------

def project_points_cpu(pts_xyz, R, t, fx, fy, cx, cy, W, H):
    x_cam = pts_xyz @ R.T + t  # (N, 3)
    z = x_cam[:, 2]
    valid = z > 0.1

    u = np.full(len(pts_xyz), -1.0)
    v = np.full(len(pts_xyz), -1.0)
    u[valid] = fx * x_cam[valid, 0] / z[valid] + cx
    v[valid] = fy * x_cam[valid, 1] / z[valid] + cy

    in_bounds = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u, v, z, in_bounds


def apply_ground_white_cpu(canvas, covered, R, fx, fy, cx, cy, W, H):
    """Set background pixels whose world-space ray points downward (Z < 0) to white."""
    uu_grid, vv_grid = np.meshgrid(np.arange(W, dtype=np.float32),
                                   np.arange(H, dtype=np.float32))  # (H, W)
    d_cam = np.stack([(uu_grid - cx) / fx,
                      (vv_grid - cy) / fy,
                      np.ones((H, W), dtype=np.float32)], axis=-1)  # (H, W, 3)
    # d_world = R.T @ d_cam  →  row-vector form: d_cam @ R
    d_world = d_cam @ R  # (H, W, 3)
    looking_down = d_world[..., 2] < 0  # Z-up: negative Z means downward
    canvas[~covered & looking_down] = 255


def render_image_cpu(pts_xyz, colors, R, t, cam, args):
    W, H = cam.width, cam.height
    fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy

    bg = 0 if args.bg_color == "black" else 255
    canvas = np.full((H, W, 3), bg, dtype=np.uint8)
    covered = np.zeros((H, W), dtype=bool)

    u, v, z, mask = project_points_cpu(pts_xyz, R, t, fx, fy, cx, cy, W, H)
    u_vis = u[mask].astype(np.int32)
    v_vis = v[mask].astype(np.int32)
    z_vis = z[mask]
    c_vis = colors[mask]

    if args.sphere:
        radii = np.clip(args.sphere_scale / z_vis, 1, args.max_radius).astype(np.int32)
        order = np.argsort(z_vis)[::-1]
        for idx in order:
            color = (int(c_vis[idx, 0]), int(c_vis[idx, 1]), int(c_vis[idx, 2]))
            cx_ = int(u_vis[idx])
            cy_ = int(v_vis[idx])
            r = int(radii[idx])
            cv2.circle(canvas, (cx_, cy_), r, color, -1)
            cv2.circle(covered.view(np.uint8), (cx_, cy_), r, 1, -1)
    elif args.point_size <= 1:
        canvas[v_vis, u_vis] = c_vis
        covered[v_vis, u_vis] = True
    else:
        r = args.point_size
        for du in range(-r, r + 1):
            for dv in range(-r, r + 1):
                if du * du + dv * dv > r * r:
                    continue
                vv = np.clip(v_vis + dv, 0, H - 1)
                uu = np.clip(u_vis + du, 0, W - 1)
                canvas[vv, uu] = c_vis
                covered[vv, uu] = True

    if args.ground_white:
        apply_ground_white_cpu(canvas, covered, R, fx, fy, cx, cy, W, H)

    return canvas


# ---------------------------------------------------------------------------
# GPU rendering (PyTorch)
# ---------------------------------------------------------------------------

def apply_ground_white_gpu(canvas, covered, R_gpu, fx, fy, cx, cy, W, H, device):
    import torch
    u_grid = torch.arange(W, device=device, dtype=torch.float32)
    v_grid = torch.arange(H, device=device, dtype=torch.float32)
    vv_grid, uu_grid = torch.meshgrid(v_grid, u_grid, indexing="ij")  # (H, W)
    d_cam = torch.stack([(uu_grid - cx) / fx,
                         (vv_grid - cy) / fy,
                         torch.ones(H, W, device=device)], dim=-1)  # (H, W, 3)
    # d_world = R.T @ d_cam  →  row-vector form: d_cam @ R
    d_world = d_cam @ R_gpu  # (H, W, 3)
    looking_down = d_world[..., 2] < 0
    canvas[~covered & looking_down] = 255


def render_image_gpu(pts, cols, R, t, cam, args):
    """pts, cols are already on GPU as torch tensors."""
    import torch
    device = pts.device

    W, H = cam.width, cam.height
    fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy

    # Only R and t are transferred per frame (tiny)
    R_gpu = torch.from_numpy(R).float().to(device)
    t_gpu = torch.from_numpy(t).float().to(device)

    # Project
    x_cam = pts @ R_gpu.T + t_gpu                            # (N, 3)
    z = x_cam[:, 2]
    valid = z > 0.1
    x_cam, z, cols_v = x_cam[valid], z[valid], cols[valid]

    u = fx * x_cam[:, 0] / z + cx
    v = fy * x_cam[:, 1] / z + cy
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[in_bounds].long()
    v = v[in_bounds].long()
    z = z[in_bounds]
    cols_v = cols_v[in_bounds]

    bg = 0 if args.bg_color == "black" else 255
    canvas = torch.full((H, W, 3), bg, dtype=torch.uint8, device=device)
    covered = torch.zeros(H, W, dtype=torch.bool, device=device)

    if args.sphere:
        radii = torch.clamp(args.sphere_scale / z, 1, args.max_radius).long()

        # Sort far-to-near: for repeated pixel writes, the closest (last) wins
        order = torch.argsort(z, descending=True)
        u, v, z, radii, cols_v = u[order], v[order], z[order], radii[order], cols_v[order]

        max_r = int(radii.max().item())

        # Loop over offsets (small: ~pi*max_r^2 iters), vectorize over points
        for du in range(-max_r, max_r + 1):
            for dv in range(-max_r, max_r + 1):
                dist2 = du * du + dv * dv
                within = dist2 <= radii ** 2
                if not within.any():
                    continue
                uu = torch.clamp(u[within] + du, 0, W - 1)
                vv = torch.clamp(v[within] + dv, 0, H - 1)
                canvas[vv, uu] = cols_v[within]
                covered[vv, uu] = True
    elif args.point_size <= 1:
        canvas[v, u] = cols_v
        covered[v, u] = True
    else:
        r = args.point_size
        for du in range(-r, r + 1):
            for dv in range(-r, r + 1):
                if du * du + dv * dv > r * r:
                    continue
                vv = torch.clamp(v + dv, 0, H - 1)
                uu = torch.clamp(u + du, 0, W - 1)
                canvas[vv, uu] = cols_v
                covered[vv, uu] = True

    if args.ground_white:
        apply_ground_white_gpu(canvas, covered, R_gpu, fx, fy, cx, cy, W, H, device)

    return canvas.cpu().numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.gpu:
        import torch
        if not torch.cuda.is_available():
            print("WARNING: --gpu requested but CUDA not available, falling back to CPU")
            args.gpu = False

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading COLMAP reconstruction...")
    sm = pycolmap.SceneManager(args.sparse_dir)
    sm.load_cameras()
    sm.load_images()
    sm.load_points3D()

    pts_xyz = sm.points3D
    pts_color = sm.point3D_colors  # (N, 3) uint8 RGB

    print(f"Points: {len(pts_xyz):,}")
    print(f"Total images: {len(sm.images)}")
    print(f"Renderer: {'GPU (CUDA)' if args.gpu else 'CPU'}")

    images = list(sm.images.values())
    if args.camera_id is not None:
        images = [img for img in images if img.camera_id == args.camera_id]
        print(f"Images for camera {args.camera_id}: {len(images)}")
    images = images[::args.step]
    print(f"Images to render (step={args.step}): {len(images)}")

    if args.gpu:
        import torch
        device = torch.device("cuda")
        # Upload point cloud to GPU once — reused for every frame
        pts_gpu = torch.from_numpy(pts_xyz).float().to(device)
        cols_gpu = torch.from_numpy(pts_color).to(device)
        render_fn = lambda xyz, col, R, t, cam, args: render_image_gpu(pts_gpu, cols_gpu, R, t, cam, args)
    else:
        render_fn = render_image_cpu

    for img in tqdm(images, desc="Rendering"):
        cam = sm.cameras[img.camera_id]
        R = img.R()
        t = img.t

        canvas = render_fn(pts_xyz, pts_color, R, t, cam, args)

        name = img.name.replace(".jpg", "").replace(".png", "")
        out_path = os.path.join(args.output_dir, f"{name}.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        Image.fromarray(canvas).save(out_path)

    print(f"\nDone. Output in: {args.output_dir}")


if __name__ == "__main__":
    main()
