#!/usr/bin/env python3
"""Convert mask format and rename files.

Features:
1. Invert mask values: 255 -> 0, non-255 -> 255
2. Rename files in-place: remove "view" from filename (e.g., xxx_view0.png -> xxx_0.png)

Usage:
    # Invert masks
    python script/convert_mask.py --input_dir data/masks --output_dir data/masks_inverted

    # Invert masks + rename (remove "view")
    python script/convert_mask.py --input_dir data/masks --output_dir data/masks_inverted --remove_view

    # Only rename in-place (no invert, no output_dir needed)
    python script/convert_mask.py --input_dir data/masks --rename_only
"""

import argparse
import os

import numpy as np
from PIL import Image
from tqdm import tqdm


def find_png_files(input_dir):
    """Recursively find all PNG files in input directory."""
    png_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith('.png'):
                png_files.append(os.path.join(root, f))
    return png_files


def convert_mask(input_path, output_path, invert=True):
    """Convert mask: optionally invert 255 <-> non-255."""
    img = Image.open(input_path)
    arr = np.array(img)

    # Handle both grayscale and RGB masks
    if arr.ndim == 3:
        # For RGB, use first channel or convert to grayscale
        arr = arr[:, :, 0]

    if invert:
        # Invert: 255 -> 0, non-255 -> 255
        arr = np.where(arr == 255, 0, 255).astype(np.uint8)

    # Save as grayscale PNG
    new_img = Image.fromarray(arr, mode='L')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    new_img.save(output_path)


def remove_view_from_filename(filename):
    """Remove 'view' from filename: xxx_view0.png -> xxx_0.png"""
    return filename.replace("view", "")


def rename_files_inplace(input_dir):
    """Rename files in-place: remove 'view' from filenames."""
    png_files = find_png_files(input_dir)
    print(f"Found {len(png_files)} PNG files in {input_dir}")

    if len(png_files) == 0:
        print("No PNG files found!")
        return

    renamed_count = 0
    for input_path in tqdm(png_files, desc="Renaming files"):
        dir_part = os.path.dirname(input_path)
        file_part = os.path.basename(input_path)
        new_file_part = remove_view_from_filename(file_part)

        if new_file_part != file_part:
            new_path = os.path.join(dir_part, new_file_part)
            os.rename(input_path, new_path)
            renamed_count += 1

    print(f"Done. Renamed {renamed_count} files.")


def main():
    p = argparse.ArgumentParser(
        description="Convert mask format and rename files"
    )
    p.add_argument("--input_dir", type=str, required=True,
                   help="Input directory containing PNG mask files")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory for converted masks (not needed for --rename_only)")
    p.add_argument("--invert", action=argparse.BooleanOptionalAction, default=True,
                   help="Invert mask values: 255->0, non-255->255 (default: True)")
    p.add_argument("--remove_view", action="store_true", default=False,
                   help="Remove 'view' from output filenames when converting")
    p.add_argument("--rename_only", action="store_true", default=False,
                   help="Only rename files in-place (remove 'view'), no mask conversion")
    args = p.parse_args()

    # Rename-only mode: just rename files in place and exit
    if args.rename_only:
        rename_files_inplace(args.input_dir)
        return

    # Normal mode: need output_dir
    if args.output_dir is None:
        p.error("--output_dir is required (unless using --rename_only)")

    # Find all PNG files
    png_files = find_png_files(args.input_dir)
    print(f"Found {len(png_files)} PNG files in {args.input_dir}")

    if len(png_files) == 0:
        print("No PNG files found!")
        return

    # Process each file
    desc = "Processing masks"
    if args.invert:
        desc += " (invert"
    if args.remove_view:
        desc += "+rename" if args.invert else " (rename"
    desc += ")" if args.invert or args.remove_view else ""

    for input_path in tqdm(png_files, desc=desc):
        # Compute relative path to maintain folder structure
        rel_path = os.path.relpath(input_path, args.input_dir)

        # Remove "view" from filename if requested
        if args.remove_view:
            dir_part = os.path.dirname(rel_path)
            file_part = os.path.basename(rel_path)
            file_part = remove_view_from_filename(file_part)
            rel_path = os.path.join(dir_part, file_part)

        output_path = os.path.join(args.output_dir, rel_path)

        convert_mask(input_path, output_path, invert=args.invert)

    print(f"Done. Output saved to {args.output_dir}")


if __name__ == "__main__":
    main()
