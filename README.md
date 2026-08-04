# gsplat

[![Core Tests.](https://github.com/nerfstudio-project/gsplat/actions/workflows/core_tests.yml/badge.svg?branch=main)](https://github.com/nerfstudio-project/gsplat/actions/workflows/core_tests.yml)
[![Docs](https://github.com/nerfstudio-project/gsplat/actions/workflows/doc.yml/badge.svg?branch=main)](https://github.com/nerfstudio-project/gsplat/actions/workflows/doc.yml)

[http://www.gsplat.studio/](http://www.gsplat.studio/)

gsplat is an open-source library for CUDA accelerated rasterization of gaussians with python bindings. It is inspired by the SIGGRAPH paper [3D Gaussian Splatting for Real-Time Rendering of Radiance Fields](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), but we’ve made gsplat even faster, more memory efficient, and with a growing list of new features! 

<div align="center">
  <video src="https://github.com/nerfstudio-project/gsplat/assets/10151885/64c2e9ca-a9a6-4c7e-8d6f-47eeacd15159" width="100%" />
</div>

## News

[May 2025] Arbitrary batching (over multiple scenes and multiple viewpoints) is supported now!! Checkout [here](docs/batch.md) for more details! Kudos to [Junchen Liu](https://junchenliu77.github.io/).

[May 2025] [Jonathan Stephens](https://x.com/jonstephens85) makes a great [tutorial video](https://www.youtube.com/watch?v=ACPTiP98Pf8) for Windows users on how to install gsplat and get start with 3DGUT.

[April 2025] [NVIDIA 3DGUT](https://research.nvidia.com/labs/toronto-ai/3DGUT/) is now integrated in gsplat! Checkout [here](docs/3dgut.md) for more details. [[NVIDIA Tech Blog]](https://developer.nvidia.com/blog/revolutionizing-neural-reconstruction-and-rendering-in-gsplat-with-3dgut/) [[NVIDIA Sweepstakes]](https://www.nvidia.com/en-us/research/3dgut-sweepstakes/)

## Installation

**Dependence**: Please install [Pytorch](https://pytorch.org/get-started/locally/) first.

The easiest way is to install from PyPI. In this way it will build the CUDA code **on the first run** (JIT).

```bash
pip install gsplat
```

Alternatively you can install gsplat from source. In this way it will build the CUDA code during installation.

```bash
pip install git+https://github.com/nerfstudio-project/gsplat.git
```

We also provide [pre-compiled wheels](https://docs.gsplat.studio/whl) for both linux and windows on certain python-torch-CUDA combinations (please check first which versions are supported). Note this way you would have to manually install [gsplat's dependencies](https://github.com/nerfstudio-project/gsplat/blob/6022cf45a19ee307803aaf1f19d407befad2a033/setup.py#L115). For example, to install gsplat for pytorch 2.0 and cuda 11.8 you can run
```
pip install ninja numpy jaxtyping rich
pip install gsplat --index-url https://docs.gsplat.studio/whl/pt20cu118
```

To build gsplat from source on Windows, please check [this instruction](docs/INSTALL_WIN.md).

## Evaluation

This repo comes with a standalone script that reproduces the official Gaussian Splatting with exactly the same performance on PSNR, SSIM, LPIPS, and converged number of Gaussians. Powered by gsplat’s efficient CUDA implementation, the training takes up to **4x less GPU memory** with up to **15% less time** to finish than the official implementation. Full report can be found [here](https://docs.gsplat.studio/main/tests/eval.html).

```bash
cd examples
pip install -r requirements.txt
# download mipnerf_360 benchmark data
python datasets/download_dataset.py
# run batch evaluation
bash benchmarks/basic.sh
```

## Examples

We provide a set of examples to get you started! Below you can find the details about
the examples (requires to install some exta dependencies via `pip install -r examples/requirements.txt`)

- [Train a 3D Gaussian splatting model on a COLMAP capture.](https://docs.gsplat.studio/main/examples/colmap.html)
- [Fit a 2D image with 3D Gaussians.](https://docs.gsplat.studio/main/examples/image.html)
- [Render a large scene in real-time.](https://docs.gsplat.studio/main/examples/large_scale.html)

## Forest / Orchard Pipeline (`script/`)

`script/` holds a self-contained pipeline built on top of the `gsplat` core for
large outdoor (forest/orchard) captures: LAS/COLMAP data prep, sky+ground
masking, block-wise multi-GPU training, and per-tree segmentation/export. All
scripts are run from the `script/` directory. Paths below are illustrative —
substitute your own `data_dir` / `result_dir`.

### 1. Data preparation

Convert a LAS point cloud (or merge point clouds) into COLMAP's
`points3D.txt`/PLY, optionally padding the sky with a synthetic hemisphere of
points so far-away sky has something to reconstruct against:

```bash
# LAS -> PLY + points3D.txt, subsampled 1/N
python convert_las.py --input data/scene.las --output_dir data/LAS --subsample 1

# Same, plus a synthetic sky hemisphere (helps sky-region reconstruction)
python convert_las.py --input data/scene.las --output_dir data/LAS/LAS20x_sky250r001 \
    --subsample 20 --add_sky --sky_radius 250 --sky_ratio 0.01

# Merge two points3D.txt (e.g. ground LiDAR + synthetic sky) into one sparse model
python convert_las.py --merge data/LAS/points3D.txt data/sparse_ori/0/points3D.txt \
    --output_dir data/sparse/0

# points3D.txt/bin -> BIN (COLMAP model_converter, for tools that require binary)
colmap model_converter --input_path data/sparse/0 --output_path data/sparse/0 --output_type BIN
```

For scenes too large to train in one shot, split `sparse/0` into 4 XY
quadrants (cameras split by mean position; `cameras.bin`/`points3D.bin` are
shared via symlink) for block-wise training (see §3):

```bash
python split_colmap.py --input data/sparse/0 --output data/sparse_blocks
```

Photographer/rig self-occlusion masks:

```bash
# Invert mask convention (255->0) and/or strip "view" from filenames
python convert_mask.py --input_dir data/masks_ori --output_dir data/masks
```

### 2. Sky / ground mask generation

Sky and ground pixels get down-weighted during training (see §3) instead of
being hard-masked out, since a soft weight still lets far structures (branch
silhouettes against the sky, ground contact points) contribute a little.
Masks are produced by rendering the sparse point cloud into each camera view
and classifying pixels by alpha, then cleaning up with connected-component
filtering:

```bash
# 1. Project the (optionally ground-removed) point cloud into every camera view.
#    --ground_white marks downward-looking empty pixels as a ground placeholder
#    (alpha=127); --remove_ground additionally CSF-filters ground points out of
#    the point cloud before projecting, so only sky/structure remain as "real" points.
python project_pointcloud.py \
    --sparse_dir data/sparse_LAS/0 \
    --output_dir data/sparse_LAS/projected_full_woGround \
    --step 1 --sphere --sphere_scale 5.0 --max_radius 50 \
    --ground_white --remove_ground --gpu

# 2. Turn the alpha-channel renders into a 3-class mask (255=sky, 127=ground, 0=other).
#    --save_intermediate dumps the pre-CC-filter mask for debugging.
python make_sky_mask.py \
    --input_dir  data/sparse_LAS/projected_full_woGround/map_0 \
    --output_dir data/sparse_LAS/projected_full_woGround/sky_mask \
    --morph_n 9 \
    --orig_dir data/images/map_0 \
    --save_intermediate
```

`mask_process.py` is a small interactive/batch tool (edit the `RUN_MODE`
macro at the top of the file) for manually blacking out a fixed image region
(e.g. a tripod/rig visible in every frame) below or between clicked lines on
a reference image, then propagating that region to every frame from the same
sub-camera.

### 3. Training (`simple_trainer.py`)

Thin wrapper around the `examples/simple_trainer.py` trainer with COLMAP
photographer/sky/ground masks, sky-depth regularization (push sky
Gaussians far away instead of letting them collapse near the camera), and
block-wise / multi-GPU support baked in. `sky_mask`/`masks` folders are
auto-detected as `<data_dir>/sky_mask` and `<data_dir>/masks` if present.

```bash
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
    --data_dir /path/to/data/perspective \
    --data_factor 1 \
    --result_dir /path/to/results/my_scene \
    --steps_scaler 20.0 \
    --pose_opt \
    --test_every 0 \
    --sh_degree 2 \
    --global_scale 0.3 \
    --sky_depth_reg 0.1 \
    --vis_every 2000 \
    --strategy.grow_grad2d 0.00002 \
    --strategy.grow_scale3d 0.002 \
    --scale_reg 0.01 \
    --init_scale 0.3 \
    --use_bilateral_grid --bilateral_grid_shape 1 1 4 \
    --strategy.max_gs 14000000
```

Notes on frequently-tuned flags (see `script/cmd.txt` for more real-world
combinations): `steps_scaler` scales all step counts (useful when
`data_factor`/dataset size changes); `strategy.grow_grad2d` /
`strategy.grow_scale3d` control Gaussian split/duplicate sensitivity (lower
`grow_grad2d` densifies more aggressively); `sky_depth_reg` penalizes sky
pixels whose rendered depth is below `sky_depth_min` (auto-derived from the
point cloud's Z range); `--resume <ckpt.pt>` continues training from a
checkpoint; `--sparse_folder sparse_blocks/sparse_00` trains a single block
(run once per block, per GPU, for block-wise scenes).

For a block-wise run, train each `sparse_XX` block independently (as above)
then merge:

```bash
python merge_blocks.py \
    --blocks_dir /path/to/results/my_scene_blocks \
    --output_dir /path/to/results/my_scene_blocks/merge \
    --split_info /path/to/data/sparse_blocks/split_info.json
```

For a multi-GPU (DDP) run on a single (non-block) scene, each rank writes its
own `ckpt_{step}_rank{r}.pt`; merge them into one checkpoint before
eval/export/viewing (optionally dropping sky Gaussians and re-clamping huge
scales left over from opacity resets):

```bash
python merge_rank_ckpts.py --ckpt_dir /path/to/results/my_scene/ckpts \
    --sparse_dir /path/to/data/sparse --remove_sky --clamp_scale 1
```

### 4. Export, format conversion & viewing

```bash
# Export: standard 3DGS PLY / re-init points3D.txt / camera poses / treeiso input
python export_ckpt.py --ckpt results/my_scene/ckpts/ckpt_merged.pt --export_splat_ply
python export_ckpt.py --ckpt results/my_scene/ckpts/ckpt_merged.pt --data_dir data/my_scene --export_poses
python export_ckpt.py --ckpt results/my_scene/ckpts/ckpt_merged.pt --export_treeiso_points

# Reverse direction: import a PLY (e.g. from another 3DGS tool) as a gsplat checkpoint,
# applying the same COLMAP normalization used during training so it lines up with it
python ply_to_ckpt.py --ply point_cloud.ply --output merge.pt \
    --data_dir data/my_scene --step 599999

# Interactive viewer, or batch-render the train/test split to disk
python simple_viewer.py --ckpt results/my_scene/ckpts/ckpt_merged.pt --port 8080
python simple_viewer.py --ckpt results/my_scene/ckpts/ckpt_merged.pt \
    --data_dir data/my_scene --output_dir results/my_scene/renders \
    --render_dataset --render_split train

# PSNR/SSIM/LPIPS from [GT|RESULT|DEPTH] concatenated comparison renders
python compute_metrics.py --input_dir results/my_scene/renders --step 8 --grayscale
```

### 5. Per-tree segmentation, labeling & rendering

Pipeline for isolating and inspecting a single tree out of a trained forest
checkpoint (builds on external `treeiso` for point-cloud tree instance
segmentation):

```bash
# 1. Export an opacity/sky-filtered point cloud for treeiso instance segmentation
python export_ckpt.py --ckpt results/forest/ckpts/ckpt_merged.pt --export_treeiso_points
# ... run pc_preprocess.py + treeiso + pc_postprocess.py --backproject externally ...

# 2. Map the per-point treeiso labels (tree ID / -1=ground / -2=noise) back onto
#    every Gaussian (label -3 = not exported / filtered out in step 1)
python assign_tree_labels.py --ckpt results/forest/ckpts/ckpt_merged.pt \
    --backproj treeiso_output/treeiso_input_clean_treeiso_backproj.npy

# 3. Extract one tree's surface point cloud (renders synthetic views around the
#    tree, back-projects depth, removes outliers)
python extract_tree_pointcloud.py \
    --ckpt results/forest/ckpts/ckpt_merged_labeled.pt --label 378 --debug --metric

# 4. Render a 360 deg orbit around a single-tree checkpoint (also dumps cameras.json,
#    used as input to the manual labeling step below)
python render_tree_orbit.py --ckpt results/forest/tree378.pt \
    --height 2 --lookat_height 3 --dist 6 --min_opacity 0.1 \
    --save_cameras --resolution 720 1280

# 5. Manually annotate one rendered orbit frame in LabelMe, then recolour the
#    Gaussians whose projections fall inside the polygon (e.g. to highlight a
#    defect/region and optionally estimate its surface area)
python label_gaussians.py \
    --ckpt results/forest/tree378.pt \
    --cameras results/forest/tree378_render_1024x1024/cameras.json \
    --annotation results/forest/tree378_render_1024x1024/frames/0006.json \
    --output results/forest/tree378_highlighted.pt \
    --compute_area --debug
```

`stitch_renders.py` stitches two directories of renders side by side (by
matching the trailing frame number in each filename) for side-by-side
comparison videos/figures.

More real-world flag combinations (multi-GPU forest runs, resuming, block
training, etc.) live in `script/cmd.txt`.

## Development and Contribution

This repository was born from the curiosity of people on the Nerfstudio team trying to understand a new rendering technique. We welcome contributions of any kind and are open to feedback, bug-reports, and improvements to help expand the capabilities of this software.

This project is developed by the following wonderful contributors (unordered):

- [Angjoo Kanazawa](https://people.eecs.berkeley.edu/~kanazawa/) (UC Berkeley): Mentor of the project.
- [Matthew Tancik](https://www.matthewtancik.com/about-me) (Luma AI): Mentor of the project.
- [Vickie Ye](https://people.eecs.berkeley.edu/~vye/) (UC Berkeley): Project lead. v0.1 lead.
- [Matias Turkulainen](https://maturk.github.io/) (Aalto University): Core developer.
- [Ruilong Li](https://www.liruilong.cn/) (UC Berkeley): Core developer. v1.0 lead.
- [Justin Kerr](https://kerrj.github.io/) (UC Berkeley): Core developer.
- [Brent Yi](https://github.com/brentyi) (UC Berkeley): Core developer.
- [Zhuoyang Pan](https://panzhy.com/) (ShanghaiTech University): Core developer.
- [Jianbo Ye](http://www.jianboye.org/) (Amazon): Core developer.

We also have a white paper with about the project with benchmarking and mathematical supplement with conventions and derivations, available [here](https://arxiv.org/abs/2409.06765). If you find this library useful in your projects or papers, please consider citing:

```
@article{ye2025gsplat,
  title={gsplat: An open-source library for Gaussian splatting},
  author={Ye, Vickie and Li, Ruilong and Kerr, Justin and Turkulainen, Matias and Yi, Brent and Pan, Zhuoyang and Seiskari, Otto and Ye, Jianbo and Hu, Jeffrey and Tancik, Matthew and Angjoo Kanazawa},
  journal={Journal of Machine Learning Research},
  volume={26},
  number={34},
  pages={1--17},
  year={2025}
}
```

We welcome contributions of any kind and are open to feedback, bug-reports, and improvements to help expand the capabilities of this software. Please check [docs/DEV.md](docs/DEV.md) for more info about development.
