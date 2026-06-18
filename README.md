# Mono to Stereo Image Conversion

Convert a single (mono) image into a stereo pair suitable for VR headsets using depth-based view synthesis and inpainting. The pipeline generates left and right views from a monocular image, fills disoccluded regions with a fine-tuned Stable Diffusion inpainting model, and combines them into a side-by-side stereo output.

## Pipeline Overview

```
Input Image
    │
    ▼
Step 1: Depth Estimation (Apple DepthPro)
    │
    ▼
Step 2: View Synthesis — R_MASK.py / L_MASK.py
    │  (forward-warp using disparity → creates masked left/right views)
    ▼
Step 3: Inpainting (Fine-tuned Stable Diffusion)
    │  (fills disoccluded holes with contextually correct content)
    ▼
Step 4: Edge-safe Refinement — align_replace_edgesafe.py
    │  (aligns inpainted output with original, feathers edges)
    ▼
Step 5: Side-by-Side Merge — align_side_by_side.py
    │
    ▼
Stereo SBS Output (ready for VR headset)
```

## Repository Structure

```
.
├── L_MASK.py                        # Left-view synthesis via two-pass forward warp
├── R_MASK.py                        # Right-view synthesis via two-pass forward warp
├── inpaint_inference.py             # Custom inpainting inference script (replaces original)
├── inpainting_example_overfit.yaml  # Model config for inpainting
├── align_replace_edgesafe.py        # Edge-safe alignment and compositing
├── align_side_by_side.py            # Merges left/right views into SBS stereo
├── rename_resize.py                 # Utility: rename/resize images before processing
├── Left/
│   └── left.ckpt                    # Fine-tuned inpainting checkpoint (left view)
└── Right/
    └── right.ckpt                   # Fine-tuned inpainting checkpoint (right view)
```

## Pre-trained Models

The fine-tuned inpainting checkpoints are hosted on HuggingFace:

👉 **[EE-AMoiz/Mono-to-Stereo-Image-Conversion](https://huggingface.co/EE-AMoiz/Mono-to-Stereo-Image-Conversion)**

Download `left.ckpt` and `right.ckpt` and place them in the `Left/` and `Right/` folders respectively before running Step 3.

## Requirements

- Python 3.8+
- `opencv-python`, `numpy`, `tqdm`
- CUDA-capable GPU (for inpainting step)
- [Apple DepthPro](https://github.com/apple/ml-depth-pro) (Step 1)
- [Stable Diffusion Inpaint](https://github.com/lorenzo-stacchio/Stable-Diffusion-Inpaint) (Step 3)

## Step-by-Step Usage

### Step 1 — Depth Estimation

Clone and install Apple's DepthPro model:

```bash
git clone https://github.com/apple/ml-depth-pro.git
cd ml-depth-pro
# follow their installation instructions
```

Run depth estimation on your input images:

```bash
depth-pro-run -i "/path/to/input/images" -o "/path/to/depth/output" --skip-display
```

This produces disparity maps for each input image.

---

### Step 2 — View Synthesis

Use `R_MASK.py` to generate the **right** view, or `L_MASK.py` for the **left** view. Both scripts perform a two-pass forward warp using the disparity maps and produce three outputs per image: the original, the masked/warped view, and the hole mask.

```bash
python R_MASK.py \
  --orig  "/path/to/original/images" \
  --disp  "/path/to/disparity/maps" \
  --output "/path/to/output" \
  --scale-factor 6 \
  --translation 1
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--scale-factor` | 6.0 | Controls shift magnitude. Higher = smaller shifts. |
| `--translation` | 2.0 | Controls left/right translation strength. |
| `--mask-dilate` | 0 | Dilates the hole mask to cover border artifacts. |
| `--workers` | auto | Number of parallel worker processes. |

The output folder will contain three subfolders: `orig/`, `masked_image/`, and `mask/`.

---

### Step 3 — Inpainting

Clone the Stable Diffusion inpainting repository:

```bash
git clone https://github.com/lorenzo-stacchio/Stable-Diffusion-Inpaint.git
```

**Replace** the original `inpaint_inference.py` with the one from this repo. Place the checkpoint from the `Left/` or `Right/` folder into `logs/checkpoints/`.

Run inpainting:

```bash
# For right view
python inpaint_inference.py \
  --indir  ./dataset/mask \
  --outdir ./dataset/image \
  --ckpt   logs/checkpoints/right.ckpt \
  --yaml_profile configs/latent-diffusion/inpainting_example_overfit.yaml \
  --device cuda \
  --steps  50 \
  --prefix finetuned \
  --ema
```

Use `left.ckpt` when generating the left view.

---

### Step 4 — Edge-safe Refinement

Align the inpainted output back to the original masked image and composite them with feathered edges to eliminate seams:

```bash
python align_replace_edgesafe.py \
  --img1_dir ./masked_image \
  --img2_dir ./model_output \
  --out_dir  ./refined_output \
  --align    translation \
  --black_thresh 10 \
  --erode    3 \
  --feather  4
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--align` | `translation` | ECC alignment mode: `translation` or `affine`. |
| `--black_thresh` | 10 | Luma threshold below which pixels are treated as holes. |
| `--erode` | 3 | Erodes mask boundary to remove contaminated rim pixels. |
| `--feather` | 4 | Feather width (px) for smooth edge blending. |

---

### Step 5 — Side-by-Side Merge

Combine the refined left and right views into a single side-by-side stereo image for VR headsets:

```bash
# Single image pair
python align_side_by_side.py ./left_image.png ./right_image.png vr_sbs.png

# Batch (folders)
python align_side_by_side.py ./left_folder ./right_folder ./output_folder
```

The output is a horizontally concatenated `[Left | Right]` image at matched resolution.
