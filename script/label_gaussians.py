#!/usr/bin/env python3
"""Recolour labelled Gaussians to a solid colour.

Projects all Gaussian means onto the annotated frame, finds those whose
projections fall inside the LabelMe polygons, sets their SH DC coefficient
to the target colour (shN zeroed out), and saves a new .pt checkpoint.

Usage:
    python script/label_gaussians.py \\
        --ckpt       path/to/tree.pt \\
        --cameras    path/to/cameras.json \\
        --annotation path/to/0005.json \\
        --output     path/to/tree_highlighted.pt
"""

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from gsplat.rendering import rasterization

# SH DC coefficient: rgb = sh0 * C0 + 0.5
_C0 = 0.28209479177387814


def rgb_to_sh0(rgb: np.ndarray) -> np.ndarray:
    """Convert linear RGB [0,1]^3 to SH DC coefficient."""
    return (np.asarray(rgb, dtype=np.float32) - 0.5) / _C0


def build_annotation_mask(shapes: list, height: int, width: int) -> np.ndarray:
    """Rasterise all LabelMe shapes into a binary uint8 mask (H, W)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for shape in shapes:
        pts = np.array(shape["points"], dtype=np.float32)
        stype = shape.get("shape_type", "polygon")
        if stype == "rectangle":
            # LabelMe rectangle: two corner points
            x0, y0 = pts[0]
            x1, y1 = pts[1]
            pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
        poly = pts.round().astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [poly], color=1)
    return mask


def project_means(means: np.ndarray, c2w: np.ndarray, K: np.ndarray):
    """Project (N,3) world-space means to pixel coords using OpenCV convention.

    Returns:
        uv      : (N, 2) float pixel coordinates
        cam_z   : (N,)  camera-space Z depth of each Gaussian centre
        valid   : (N,)  bool — True when point is in front of the camera
    """
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    pts_cam = means @ R_inv.T + t_inv          # (N, 3)

    cam_z = pts_cam[:, 2]
    valid = cam_z > 0
    z = pts_cam[:, 2:3]
    xy = pts_cam[:, :2] / (z + 1e-8)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = xy[:, 0] * fx + cx
    v = xy[:, 1] * fy + cy
    uv = np.stack([u, v], axis=1)
    return uv, cam_z, valid


def fit_quadric_patch(points: np.ndarray):
    """Fit a local quadric height-field (Monge patch) to a (N,3) point cloud.

    The local frame is PCA-aligned: u, v span the tangent plane (largest two
    variance directions), n is the local "normal" (smallest-variance direction).
    The surface is z = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f in that frame —
    this fits any locally smooth bend (cylindrical, conical, saddle, tapered…),
    not just a perfect circular cylinder.

    Returns:
        centroid : (3,) patch centroid
        basis    : (3,3) columns [u, v, n]
        coeffs   : (6,) [a, b, c, d, e, f]
    """
    centroid = points.mean(axis=0)
    pts = points - centroid
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)   # ascending eigenvalue order
    n = eigvecs[:, 0]
    u = eigvecs[:, 2]
    v = eigvecs[:, 1]
    x, y, z = pts @ u, pts @ v, pts @ n
    A = np.stack([x ** 2, y ** 2, x * y, x, y, np.ones_like(x)], axis=1)
    coeffs, *_ = np.linalg.lstsq(A, z, rcond=None)
    basis = np.stack([u, v, n], axis=1)
    return centroid, basis, coeffs


def eval_quadric(x: np.ndarray, y: np.ndarray, coeffs: np.ndarray):
    a, b, c, d, e, f = coeffs
    return a * x ** 2 + b * y ** 2 + c * x * y + d * x + e * y + f


def quadric_gradient(x: np.ndarray, y: np.ndarray, coeffs: np.ndarray):
    a, b, c, d, e, f = coeffs
    fx = 2 * a * x + c * y + d
    fy = 2 * b * y + c * x + e
    return fx, fy


def project_to_patch(points: np.ndarray, centroid: np.ndarray, basis: np.ndarray):
    """Project (N,3) points into the patch's local (x, y, z) frame."""
    u, v, n = basis[:, 0], basis[:, 1], basis[:, 2]
    rel = points - centroid[None, :]
    return rel @ u, rel @ v, rel @ n


def fit_quadric_patch_robust(points: np.ndarray, n_iter: int = 10, c_tukey: float = 4.685):
    """Tukey-biweight IRLS fit of the quadric patch — robust to a handful of outliers.

    A plain (unweighted) least-squares fit gets pulled off the true surface by
    even a few outliers (their squared residuals dominate the objective), which
    then makes *normal* points look anomalous too. IRLS iteratively downweights
    high-residual points so the surface itself converges to the clean data.

    Returns:
        centroid, basis, coeffs : same as fit_quadric_patch
        weights : (N,) final IRLS weight per point, ~0 for points the fit
                  effectively ignored (i.e. outliers)
    """
    centroid = points.mean(axis=0)
    pts = points - centroid
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    n = eigvecs[:, 0]
    u = eigvecs[:, 2]
    v = eigvecs[:, 1]
    x, y, z = pts @ u, pts @ v, pts @ n
    A = np.stack([x ** 2, y ** 2, x * y, x, y, np.ones_like(x)], axis=1)

    weights = np.ones_like(z)
    coeffs = np.zeros(6)
    for _ in range(n_iter):
        w_sqrt = np.sqrt(weights)
        coeffs, *_ = np.linalg.lstsq(A * w_sqrt[:, None], z * w_sqrt, rcond=None)
        resid = z - A @ coeffs
        sigma = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-12
        t = resid / (c_tukey * sigma)
        weights = np.where(np.abs(t) < 1.0, (1.0 - t ** 2) ** 2, 0.0)

    basis = np.stack([u, v, n], axis=1)
    return centroid, basis, coeffs, weights


def set_axes_equal_3d(ax):
    """Force equal aspect ratio on a 3D matplotlib axis (no native support)."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centers = limits.mean(axis=1)
    radius = 0.5 * max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def plot_patch_2d(points_2d: np.ndarray, area: float, save_path: str):
    """Scatter plot of the patch in its local tangent-plane (x, y) coordinates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(points_2d[:, 0], points_2d[:, 1], s=2)
    ax.set_aspect("equal")
    ax.set_xlabel("local x (m)")
    ax.set_ylabel("local y (m)")
    ax.set_title(f"Tangent-plane projection  area={area:.4f} m^2")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_quadric_patch_mesh(centroid: np.ndarray, basis: np.ndarray, coeffs: np.ndarray,
                             x: np.ndarray, y: np.ndarray, n_grid: int = 40):
    """Build a regular grid mesh of the fitted quadric over the points' (x, y) extent.

    Returns (verts (n_grid*n_grid, 3), faces (list of (i,j,k) triangles)).
    """
    u, v, n = basis[:, 0], basis[:, 1], basis[:, 2]
    x_grid = np.linspace(x.min(), x.max(), n_grid)
    y_grid = np.linspace(y.min(), y.max(), n_grid)
    XX, YY = np.meshgrid(x_grid, y_grid)
    ZZ = eval_quadric(XX, YY, coeffs)
    verts = (centroid[None, None, :]
             + XX[..., None] * u[None, None, :]
             + YY[..., None] * v[None, None, :]
             + ZZ[..., None] * n[None, None, :])
    verts = verts.reshape(-1, 3)

    faces = []
    for i in range(n_grid - 1):
        for j in range(n_grid - 1):
            v00 = i * n_grid + j
            v01 = i * n_grid + j + 1
            v10 = (i + 1) * n_grid + j
            v11 = (i + 1) * n_grid + j + 1
            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))
    return verts, faces


def save_obj(verts: np.ndarray, faces: list, save_path: str):
    """Write a triangle mesh to a Wavefront .obj file (1-indexed faces)."""
    with open(save_path, "w") as f:
        for vx, vy, vz in verts:
            f.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")
        for i, j, k in faces:
            f.write(f"f {i + 1} {j + 1} {k + 1}\n")


def plot_quadric_fit(kept_pts: np.ndarray, removed_pts: np.ndarray, centroid: np.ndarray,
                     basis: np.ndarray, coeffs: np.ndarray, x: np.ndarray, y: np.ndarray,
                     save_path: str):
    """3D scatter of kept/removed points plus the fitted quadric surface patch."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

    verts, _ = build_quadric_patch_mesh(centroid, basis, coeffs, x, y, n_grid=30)
    surf = verts.reshape(30, 30, 3)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(kept_pts[:, 0], kept_pts[:, 1], kept_pts[:, 2], s=2, c="red", label="kept")
    if len(removed_pts) > 0:
        ax.scatter(removed_pts[:, 0], removed_pts[:, 1], removed_pts[:, 2],
                   s=10, c="black", marker="x", label="removed outliers")
    ax.plot_surface(surf[..., 0], surf[..., 1], surf[..., 2], alpha=0.3, color="gray",
                    linewidth=0, antialiased=True)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("Quadric (Monge patch) fit")
    ax.legend()
    set_axes_equal_3d(ax)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_occupied_cells(points_2d: np.ndarray, cell_size: float = None):
    """Grid-occupancy footprint of a 2D point set. Returns (cell_centers, cell_size).

    For scattered points, a cell exactly the size of the median nearest-neighbour
    distance still leaves many cells empty by chance (Poisson-style gaps), which
    underestimates area by up to ~5x. Empirically, ~3x the median NN distance
    gives a much more stable estimate; tune via --area_cell_size if needed.
    """
    if cell_size is None:
        from scipy.spatial import cKDTree
        tree = cKDTree(points_2d)
        dists, _ = tree.query(points_2d, k=2)
        median_nn = float(np.median(dists[:, 1]))
        cell_size = max(3.0 * median_nn, 1e-6)

    cells = np.floor(points_2d / cell_size).astype(np.int64)
    unique_cells = np.unique(cells, axis=0)
    cell_centers = (unique_cells + 0.5) * cell_size
    return cell_centers, cell_size


def main():
    p = argparse.ArgumentParser(
        description="Recolour labelled Gaussians to a solid colour"
    )
    p.add_argument("--ckpt",       required=True, help="Single-tree checkpoint (.pt)")
    p.add_argument("--cameras",    required=True, help="cameras.json from render_tree_orbit.py")
    p.add_argument("--annotation", required=True, help="LabelMe JSON annotation file")
    p.add_argument("--color",      type=float, nargs=3, default=[1.0, 0.0, 0.0],
                   metavar=("R", "G", "B"),
                   help="Target colour in linear RGB [0,1] (default: 1 0 0 = red)")
    p.add_argument("--depth_thresh", type=float, default=0.02,
                   help="Keep only Gaussians whose camera-Z is within this value of the "
                        "rendered ED depth (filters back-facing Gaussians). "
                        "Set to ~10%% of the tree's depth extent as a starting point.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output",     default=None,
                   help="Output .pt path (default: <ckpt_stem>_labeled.pt)")
    p.add_argument("--compute_area", action="store_true",
                   help="Fit a local quadric surface to the selected Gaussians, remove "
                        "residual outliers, and estimate the patch's surface area.")
    p.add_argument("--area_cell_size", type=float, default=None,
                   help="Grid cell size (metres) used for area estimation on the patch's "
                        "tangent plane. Default: auto, from median nearest-neighbour distance.")
    p.add_argument("--area_outlier_thresh", type=float, default=4.685,
                   help="Tukey biweight tuning constant for the robust quadric fit used to "
                        "flag outliers (default: 4.685, the standard value giving ~95%% "
                        "efficiency under normal residuals). Lower = remove more points.")
    p.add_argument("--debug", action="store_true",
                   help="Save intermediate results (tangent-plane + quadric fit plots/mesh) "
                        "to a debug/ subfolder next to --output.")
    args = p.parse_args()

    if args.output is None:
        stem = os.path.splitext(os.path.abspath(args.ckpt))[0]
        args.output = stem + "_labeled.pt"

    debug_dir = os.path.join(os.path.dirname(os.path.abspath(args.output)), "debug")
    if args.debug:
        os.makedirs(debug_dir, exist_ok=True)

    # ── Load inputs ───────────────────────────────────────────────────────────
    print("Loading checkpoint …")
    ckpt   = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    splats = ckpt["splats"]

    if "sh0" not in splats:
        raise ValueError("Checkpoint has no sh0/shN — appearance-model checkpoints not supported.")

    with open(args.cameras) as f:
        cameras = json.load(f)   # list of {frame, c2w, K, width, height}

    with open(args.annotation) as f:
        ann = json.load(f)

    # ── Determine which frame was annotated ───────────────────────────────────
    # LabelMe stores the source image filename in imagePath, e.g. "0005.png"
    image_path = ann.get("imagePath", "")
    frame_stem = os.path.splitext(os.path.basename(image_path))[0]
    try:
        frame_idx = int(frame_stem)
    except ValueError:
        raise ValueError(
            f"Cannot parse frame index from imagePath '{image_path}'. "
            "Expected a zero-padded integer filename like '0005.png'."
        )
    print(f"Annotated frame: {frame_idx}")

    # Fetch camera for this frame
    cam_entry = next((c for c in cameras if c["frame"] == frame_idx), None)
    if cam_entry is None:
        raise ValueError(f"Frame {frame_idx} not found in {args.cameras}")

    c2w    = np.array(cam_entry["c2w"],  dtype=np.float32)  # (4, 4)
    K      = np.array(cam_entry["K"],    dtype=np.float32)  # (3, 3)
    cam_W  = cam_entry["width"]
    cam_H  = cam_entry["height"]

    # LabelMe records the dimensions of the PNG it opened.
    # If depth was rendered side-by-side, imageWidth = 2*cam_W; annotations on
    # the left (RGB) half are what we want.
    ann_W  = ann.get("imageWidth",  cam_W)
    ann_H  = ann.get("imageHeight", cam_H)

    if ann_W == cam_W * 2:
        print("Detected side-by-side RGB+depth image; using left (RGB) half.")
        # No coordinate adjustment needed: polygon points are already in the
        # RGB half as long as the user annotated the left side.
        mask_W = cam_W
    else:
        mask_W = ann_W   # should equal cam_W

    # ── Build annotation mask ─────────────────────────────────────────────────
    shapes = ann.get("shapes", [])
    if not shapes:
        raise ValueError("No shapes found in annotation JSON.")
    mask = build_annotation_mask(shapes, ann_H, mask_W)
    n_annotated_px = int(mask.sum())
    print(f"Annotation mask: {n_annotated_px:,} pixels across {len(shapes)} shape(s)")

    # ── Project Gaussian means ────────────────────────────────────────────────
    means_np = splats["means"].float().numpy()   # (N, 3)
    uv, cam_z, in_front = project_means(means_np, c2w, K)

    u_i = np.round(uv[:, 0]).astype(np.int32)
    v_i = np.round(uv[:, 1]).astype(np.int32)
    in_bounds = (u_i >= 0) & (u_i < mask_W) & (v_i >= 0) & (v_i < ann_H)

    valid = in_front & in_bounds
    in_mask = np.zeros(len(means_np), dtype=bool)
    in_mask[valid] = mask[v_i[valid], u_i[valid]] > 0

    # ── Depth filter (optional) ───────────────────────────────────────────────
    if args.depth_thresh is not None:
        device = torch.device(args.device)
        means_t     = splats["means"].float().to(device)
        quats_t     = splats["quats"].float().to(device)
        scales_t    = torch.exp(splats["scales"].float()).to(device)
        opacities_t = torch.sigmoid(splats["opacities"].float()).to(device)
        colors_t    = splats["sh0"].float().to(device)           # DC only, depth doesn't use colour

        viewmat = torch.from_numpy(
            np.linalg.inv(c2w).astype(np.float32)
        ).unsqueeze(0).to(device)
        K_t = torch.from_numpy(K).unsqueeze(0).to(device)

        with torch.no_grad():
            depth_out, _, _ = rasterization(
                means=means_t, quats=quats_t, scales=scales_t,
                opacities=opacities_t, colors=colors_t,
                viewmats=viewmat, Ks=K_t,
                width=cam_W, height=cam_H,
                render_mode="ED", sh_degree=0, rasterize_mode="classic",
            )
        ed_depth = depth_out[0, :, :, 0].cpu().numpy()          # (H, W)

        # For each candidate Gaussian, compare its camera-Z to the rendered depth
        # at its projected pixel. Keep if cam_z <= ed_depth + thresh (front-facing).
        candidates = np.where(in_mask)[0]
        depth_at_pixel = ed_depth[v_i[candidates], u_i[candidates]]
        depth_ok = cam_z[candidates] <= depth_at_pixel + args.depth_thresh
        in_mask[candidates] = depth_ok
        print(f"After depth filter (thresh={args.depth_thresh}): "
              f"{int(in_mask.sum()):,} / {len(candidates):,} candidates kept")
    else:
        print("Depth filter disabled (pass --depth_thresh to filter back-facing Gaussians)")

    n_selected = int(in_mask.sum())
    print(f"Selected Gaussians: {n_selected:,} / {len(means_np):,}")
    if n_selected == 0:
        print("WARNING: no Gaussians selected — check that camera parameters match the annotation.")
        return

    # ── Surface area estimation (optional) ───────────────────────────────────
    if args.compute_area:
        selected_pts = means_np[in_mask]
        if len(selected_pts) < 8:
            print("WARNING: too few points to fit a surface reliably; skipping area estimate.")
        else:
            # Robust (Tukey IRLS) fit to find outliers without letting them skew the
            # surface estimate, then a clean refit on the surviving inliers only.
            _, _, _, weights = fit_quadric_patch_robust(
                selected_pts, c_tukey=args.area_outlier_thresh)
            keep = weights > 0.05
            n_removed = int((~keep).sum())
            removed_pts = selected_pts[~keep]
            clean_pts = selected_pts[keep]
            if n_removed > 0:
                print(f"Denoising: removed {n_removed} / {len(selected_pts)} outliers "
                      f"(Tukey biweight c={args.area_outlier_thresh})")
            else:
                print("Denoising: no significant outliers found")
            centroid, basis, coeffs = fit_quadric_patch(clean_pts)

            x, y, z = project_to_patch(clean_pts, centroid, basis)
            points_2d = np.stack([x, y], axis=1)
            cell_centers, cell_size = compute_occupied_cells(points_2d, args.area_cell_size)

            area_flat = len(cell_centers) * cell_size ** 2
            fxg, fyg = quadric_gradient(cell_centers[:, 0], cell_centers[:, 1], coeffs)
            stretch = np.sqrt(1.0 + fxg ** 2 + fyg ** 2)
            area = float(np.sum(stretch) * cell_size ** 2)

            bbox_area = (x.max() - x.min()) * (y.max() - y.min())
            print(f"Quadric fit: coeffs(a,b,c,d,e,f)={coeffs.round(4).tolist()}")
            print(f"Tangent-plane extent: x={x.max()-x.min():.4f} m, y={y.max()-y.min():.4f} m, "
                  f"grid cell={cell_size:.4f} m")
            print(f"Bounding-box area (flat, upper bound): {bbox_area:.4f} m^2")
            print(f"Flat footprint area (no curvature correction): {area_flat:.4f} m^2")
            print(f"Estimated surface area (curvature-corrected): {area:.4f} m^2 "
                  f"({len(clean_pts):,} points)  [fill ratio vs bbox: {area_flat / max(bbox_area, 1e-9):.2f}]")

            if args.debug:
                patch2d_path = os.path.join(debug_dir, "patch_2d.png")
                quadric_path = os.path.join(debug_dir, "quadric_fit.png")
                quadric_obj_path = os.path.join(debug_dir, "quadric_fit.obj")
                plot_patch_2d(points_2d, area, patch2d_path)
                plot_quadric_fit(clean_pts, removed_pts, centroid, basis, coeffs, x, y, quadric_path)
                verts, faces = build_quadric_patch_mesh(centroid, basis, coeffs, x, y)
                save_obj(verts, faces, quadric_obj_path)
                print(f"Tangent-plane plot   → {patch2d_path}")
                print(f"Quadric fit plot     → {quadric_path}")
                print(f"Quadric fit mesh     → {quadric_obj_path}")

    # ── Recolour SH ───────────────────────────────────────────────────────────
    sh0_target = rgb_to_sh0(args.color)         # (3,)
    print(f"Target colour RGB={args.color}  →  sh0={sh0_target.tolist()}")

    sh0 = splats["sh0"].clone().float()         # (N, 1, 3)
    shN = splats["shN"].clone().float()         # (N, K, 3)

    idx = torch.from_numpy(np.where(in_mask)[0])
    sh0[idx, 0, :] = torch.tensor(sh0_target)
    shN[idx, :, :] = 0.0                        # zero out view-dependent terms

    # ── Save new checkpoint ───────────────────────────────────────────────────
    new_splats = {k: v.clone() for k, v in splats.items()}
    new_splats["sh0"] = sh0
    new_splats["shN"] = shN

    new_ckpt = {k: v for k, v in ckpt.items() if k != "splats"}
    new_ckpt["splats"] = new_splats

    torch.save(new_ckpt, args.output)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
