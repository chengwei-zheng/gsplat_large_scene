"""
Merge block-wise 3DGS training results into a single scene.

For each block directory, takes the latest checkpoint, denormalizes the splats
back to original COLMAP coordinates, then clips each block to its assigned
XY quadrant (using the split_center from split_info.json) to remove Gaussians
that drifted into neighbouring blocks during training.

Outputs:
    merge/
        merged.pt   -- merged checkpoint (splats in COLMAP coordinates)
        merged.ply  -- merged point cloud

Usage:
    python merge_blocks.py \\
        --blocks_dir /path/to/results/block_root \\
        --output_dir /path/to/results/block_root/merge \\
        --split_info /path/to/sparse_blocks/split_info.json \\
        [--block_prefix block_]
"""

import argparse
import json
import os
import re
import torch
import torch.nn.functional as F
from gsplat import export_splats


# ---------------------------------------------------------------------------
# Denormalize (mirrors Trainer._denormalize_splats)
# ---------------------------------------------------------------------------

@torch.no_grad()
def denormalize_splats(means, scales, quats, transform):
    """
    Convert splats from normalized space back to original COLMAP coordinates.

    transform: (4, 4) similarity matrix T such that P_norm = T @ [P_orig; 1]
    """
    T = transform.float()

    s = torch.linalg.norm(T[:3, 0])   # scale factor
    R = T[:3, :3] / s                  # pure rotation
    t = T[:3, 3]                       # translation

    # Inverse: P_orig = (1/s) * R^T @ (P_norm - t)
    R_inv = R.T
    means_out = (means - t[None, :]) @ R_inv.T / s

    # Log-scale: undo the scale factor
    scales_out = scales - torch.log(s)

    # Quaternions: left-multiply by quaternion of R_inv
    trace = R_inv[0, 0] + R_inv[1, 1] + R_inv[2, 2]
    if trace > 0:
        w = 0.5 * torch.sqrt(1.0 + trace)
        s4 = 0.25 / w
        x = (R_inv[2, 1] - R_inv[1, 2]) * s4
        y = (R_inv[0, 2] - R_inv[2, 0]) * s4
        z = (R_inv[1, 0] - R_inv[0, 1]) * s4
    else:
        x = y = z = torch.tensor(0.0)
        w = torch.tensor(1.0)
    q_R_inv = torch.stack([w, x, y, z]).to(means.device)

    qw, qx, qy, qz = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    rw, rx, ry, rz = q_R_inv
    out_w = rw * qw - rx * qx - ry * qy - rz * qz
    out_x = rw * qx + rx * qw + ry * qz - rz * qy
    out_y = rw * qy - rx * qz + ry * qw + rz * qx
    out_z = rw * qz + rx * qy - ry * qx + rz * qw
    quats_out = torch.stack([out_w, out_x, out_y, out_z], dim=-1)

    return means_out, scales_out, quats_out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def latest_file(directory, pattern):
    """Return the file matching pattern with the largest step number."""
    files = [f for f in os.listdir(directory) if re.match(pattern, f)]
    if not files:
        return None
    def step_of(name):
        m = re.search(r"(\d+)", name)
        return int(m.group(1)) if m else -1
    return os.path.join(directory, max(files, key=step_of))


def find_block_dirs(blocks_dir, prefix):
    """Return sorted list of (block_name, block_path) for all block dirs."""
    dirs = []
    for name in sorted(os.listdir(blocks_dir)):
        path = os.path.join(blocks_dir, name)
        if name.startswith(prefix) and os.path.isdir(path):
            dirs.append((name, path))
    return dirs


def quadrant_mask(means, quad_key, cx, cy):
    """
    Build a boolean mask selecting Gaussians that belong to this quadrant.

    quad_key is the two-character suffix from the block name (e.g. "00"):
        first char  = x-bit  (0: x < cx,  1: x >= cx)
        second char = y-bit  (0: y < cy,  1: y >= cy)
    """
    x_bit, y_bit = quad_key[0], quad_key[1]
    mx = means[:, 0]
    my = means[:, 1]
    mask_x = (mx < cx) if x_bit == "0" else (mx >= cx)
    mask_y = (my < cy) if y_bit == "0" else (my >= cy)
    return mask_x & mask_y


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Merge block-wise 3DGS results into one scene"
    )
    parser.add_argument("--blocks_dir", required=True,
                        help="Directory containing block_XX subdirectories")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for merged results")
    parser.add_argument("--split_info", required=True,
                        help="Path to split_info.json produced by split_colmap.py")
    parser.add_argument("--block_prefix", default="block_",
                        help="Prefix used for block directory names (default: block_)")
    parser.add_argument("--device", default="cpu",
                        help="Device for tensor ops (default: cpu)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Load split info
    with open(args.split_info) as f:
        split_info = json.load(f)
    cx = split_info["split_center"]["x"]
    cy = split_info["split_center"]["y"]
    print(f"Split center: x={cx:.4f}, y={cy:.4f}")

    block_dirs = find_block_dirs(args.blocks_dir, args.block_prefix)
    if not block_dirs:
        raise RuntimeError(f"No block directories found in {args.blocks_dir} "
                           f"with prefix '{args.block_prefix}'")
    print(f"Found {len(block_dirs)} blocks: {[n for n, _ in block_dirs]}")

    # -----------------------------------------------------------------------
    # Load, denormalize, and clip each block
    # -----------------------------------------------------------------------
    all_means, all_scales, all_quats = [], [], []
    all_opacities, all_sh0, all_shN = [], [], []
    block_infos = []

    for block_name, block_path in block_dirs:
        ckpt_dir = os.path.join(block_path, "ckpts")
        if not os.path.isdir(ckpt_dir):
            print(f"  [{block_name}] No ckpts/ directory, skipping")
            continue

        ckpt_path = latest_file(ckpt_dir, r"ckpt_\d+_rank0\.pt")
        if ckpt_path is None:
            print(f"  [{block_name}] No checkpoint found, skipping")
            continue

        step = int(re.search(r"ckpt_(\d+)_rank0\.pt",
                              os.path.basename(ckpt_path)).group(1))
        print(f"  [{block_name}] Loading ckpt step={step}: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        splats = ckpt["splats"]
        transform = ckpt["transform"].to(device)

        means   = splats["means"].to(device)
        scales  = splats["scales"].to(device)
        quats   = splats["quats"].to(device)
        opas    = splats["opacities"].to(device)
        sh0     = splats["sh0"].to(device)
        shN     = splats["shN"].to(device)

        means, scales, quats = denormalize_splats(means, scales, quats, transform)
        n_total = means.shape[0]

        # Clip to this block's quadrant
        # block name suffix after prefix gives the quad key, e.g. "00"
        quad_key = block_name[len(args.block_prefix):]
        if len(quad_key) == 2 and quad_key.isdigit():
            mask = quadrant_mask(means, quad_key, cx, cy)
            means  = means[mask]
            scales = scales[mask]
            quats  = quats[mask]
            opas   = opas[mask]
            sh0    = sh0[mask]
            shN    = shN[mask]
            n_kept = means.shape[0]
            print(f"    {n_total:,} -> {n_kept:,} Gaussians after quadrant clip "
                  f"(removed {n_total - n_kept:,})")
        else:
            n_kept = n_total
            print(f"    [{block_name}] Cannot parse quad key '{quad_key}', "
                  f"skipping clip. {n_total:,} Gaussians kept.")

        all_means.append(means)
        all_scales.append(scales)
        all_quats.append(quats)
        all_opacities.append(opas)
        all_sh0.append(sh0)
        all_shN.append(shN)
        block_infos.append({"block": block_name, "step": step,
                             "n_gs_raw": n_total, "n_gs": n_kept,
                             "ckpt": ckpt_path})

    if not all_means:
        raise RuntimeError("No valid blocks found to merge.")

    # -----------------------------------------------------------------------
    # Concatenate
    # -----------------------------------------------------------------------
    print("Merging ...")
    merged_means    = torch.cat(all_means,     dim=0)
    merged_scales   = torch.cat(all_scales,    dim=0)
    merged_quats    = torch.cat(all_quats,     dim=0)
    merged_opas     = torch.cat(all_opacities, dim=0)
    merged_sh0      = torch.cat(all_sh0,       dim=0)
    merged_shN      = torch.cat(all_shN,       dim=0)
    total = merged_means.shape[0]
    print(f"  Total Gaussians: {total:,}")

    # -----------------------------------------------------------------------
    # Save merged checkpoint
    # -----------------------------------------------------------------------
    ckpt_out = os.path.join(args.output_dir, "merged.pt")
    torch.save({
        "splats": {
            "means":     merged_means,
            "scales":    merged_scales,
            "quats":     merged_quats,
            "opacities": merged_opas,
            "sh0":       merged_sh0,
            "shN":       merged_shN,
        },
        # identity transform: splats are already in COLMAP coordinates
        "transform": torch.eye(4),
        "block_infos": block_infos,
        "split_center": {"x": cx, "y": cy},
    }, ckpt_out)
    print(f"  Saved merged checkpoint: {ckpt_out}")

    # -----------------------------------------------------------------------
    # Save merged ply
    # -----------------------------------------------------------------------
    ply_out = os.path.join(args.output_dir, "merged.ply")
    export_splats(
        means=merged_means,
        scales=merged_scales,
        quats=F.normalize(merged_quats, dim=-1),
        opacities=merged_opas,
        sh0=merged_sh0,
        shN=merged_shN,
        format="ply",
        save_to=ply_out,
    )
    print(f"  Saved merged ply: {ply_out}")

    print("\nDone.")
    for info in block_infos:
        print(f"  {info['block']:12s}  step={info['step']:>7d}  "
              f"gs_raw={info['n_gs_raw']:>10,}  gs_clipped={info['n_gs']:>10,}")
    print(f"  {'TOTAL':12s}  {'':>14s}  {'':>20s}  gs={total:>10,}")


if __name__ == "__main__":
    main()
