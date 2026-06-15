#!/usr/bin/env python3
"""Map treeiso back-projection labels onto the full Gaussian array.

Pipeline recap:
  1. export_ckpt.py --export_treeiso_points
       → treeiso_input.ply          (opacity/sky-filtered Gaussian means)
       → treeiso_input_index_map.npz  (survivor[i] = Gaussian index for PLY point i)
  2. pc_preprocess.py + treeiso + pc_postprocess.py --backproject
       → treeiso_input_clean_treeiso_backproj.npy
            shape (N_ply,), aligned with treeiso_input.ply
            label >= 0 : tree ID
            label == -1: ground (removed by CSF)
            label == -2: noise  (removed by SOR)
  3. This script:
       gaussian_labels[survivor] = backproj_labels
       → gaussian_labels.npy, shape (N_total_gaussians,)
            label >= 0 : tree ID
            label == -1: ground
            label == -2: noise
            label == -3: not exported (failed opacity / sky filter)

Usage:
    python script/assign_tree_labels.py \\
        --ckpt       path/to/ckpt_merged.pt \\
        --index_map  path/to/treeiso_input_index_map.npz \\
        --backproj   path/to/treeiso_input_clean_treeiso_backproj.npy
    # → saves path/to/ckpt_merged_labeled.pt with ckpt["tree_labels"] added
"""

import argparse
import os

import numpy as np
import torch

LABEL_UNSET  = -3   # Gaussian not exported (filtered out before treeiso)
LABEL_GROUND = -1   # removed by CSF ground filter
LABEL_NOISE  = -2   # removed by SOR noise filter


def main():
    p = argparse.ArgumentParser(
        description="Map treeiso back-projection labels onto the full Gaussian array"
    )
    p.add_argument("--ckpt", required=True,
                   help="Merged checkpoint .pt (used to read total Gaussian count)")
    p.add_argument("--index_map", default=None,
                   help="treeiso_input_index_map.npz produced by export_ckpt.py "
                        "(default: <ckpt_stem>/treeiso_input_index_map.npz)")
    p.add_argument("--backproj", required=True,
                   help="treeiso_input_clean_treeiso_backproj.npy produced by pc_postprocess.py")
    p.add_argument("--output_ckpt", default=None,
                   help="Output checkpoint .pt path (default: <ckpt_stem>_labeled.pt next to --ckpt)")
    args = p.parse_args()

    if args.index_map is None:
        ckpt_stem = os.path.splitext(os.path.abspath(args.ckpt))[0]
        args.index_map = os.path.join(ckpt_stem, "treeiso_input_index_map.npz")
    if not os.path.exists(args.index_map):
        raise FileNotFoundError(f"index_map not found: {args.index_map}")

    # Total Gaussian count from checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    n_gaussians = ckpt["splats"]["means"].shape[0]
    print(f"Total Gaussians: {n_gaussians:,}")

    # survivor[i] = index into Gaussian array for PLY point i
    idx_data = np.load(args.index_map)
    survivor = idx_data["survivor"]
    print(f"Exported to PLY: {len(survivor):,}")

    # per-PLY-point labels, aligned with treeiso_input.ply (= survivor space)
    backproj = np.load(args.backproj)
    if len(backproj) != len(survivor):
        raise ValueError(
            f"Size mismatch: backproj has {len(backproj)} entries "
            f"but index_map has {len(survivor)} survivors"
        )

    # Build full Gaussian label array; LABEL_UNSET for Gaussians not in the PLY
    gaussian_labels = np.full(n_gaussians, LABEL_UNSET, dtype=np.int32)
    gaussian_labels[survivor] = backproj

    n_trees     = len(np.unique(gaussian_labels[gaussian_labels >= 0]))
    n_tree_gs   = (gaussian_labels >= 0).sum()
    n_ground_gs = (gaussian_labels == LABEL_GROUND).sum()
    n_noise_gs  = (gaussian_labels == LABEL_NOISE).sum()
    n_unset_gs  = (gaussian_labels == LABEL_UNSET).sum()
    print(f"Tree Gaussians : {n_tree_gs:,} across {n_trees} trees")
    print(f"Ground         : {n_ground_gs:,}")
    print(f"Noise          : {n_noise_gs:,}")
    print(f"Unset          : {n_unset_gs:,}")

    # Write tree_labels into a new checkpoint (do not overwrite the original)
    ckpt["tree_labels"] = torch.from_numpy(gaussian_labels)

    if args.output_ckpt is None:
        ckpt_abs = os.path.abspath(args.ckpt)
        stem = os.path.splitext(ckpt_abs)[0]
        args.output_ckpt = stem + "_labeled.pt"
    torch.save(ckpt, args.output_ckpt)
    print(f"Saved: {args.output_ckpt}")


if __name__ == "__main__":
    main()
