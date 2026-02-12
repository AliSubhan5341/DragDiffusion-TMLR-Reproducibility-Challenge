### DragDiffusion Reproducibility (TMLR Reproducibility Challenge)

**Authors (Reproducibility Study)**  
- [Ali Subhan](https://www.linkedin.com/in/ali5341/), University of Ljubljana  
- [Ashir Raza](https://www.linkedin.com/in/ashir-raza7890/), University of Ljubljana  

This repository contains our reproduction and extension of **DragDiffusion: Harnessing Diffusion Models for Interactive Point-based Image Editing** as part of the **TMLR Reproducibility Challenge**.

We build on the original DragDiffusion codebase and focus on **systematic evaluation on DragBench**, including:

- **Timestep ablations** (optimized diffusion timestep \(t\))
- **LoRA training step ablations**
- Quantitative metrics:
  - **Mean Distance (MD)** between desired target points and final handle positions
  - **Image Fidelity (IF)** defined as \( \text{IF} = 1 - \text{LPIPS} \)

---

### 1. Repository Structure

Key directories and files used in our reproducibility experiments:

- `drag_bench_evaluation/`
  - `drag_bench_data/` – DragBench dataset (10 categories, original images + metadata)
  - `drag_bench_lora/` – LoRA weights trained per image
  - `run_lora_training.py` – script to train LoRA on all DragBench images
  - `run_drag_diffusion.py` – DragDiffusion runner for DragBench
  - `run_eval_point_matching.py` – computes **Mean Distance (MD)**
  - `run_eval_image_fidelity.py` – computes **IF = 1 - LPIPS**
  - `run_timestep_ablation.py` – unified script for **timestep & LoRA ablations**
  - `eval_*_ablation.sh` – helper scripts for other ablations (UNet block, lambda, etc.)
- `utils/`
  - `lora_utils.py` – LoRA training logic (used by `run_lora_training.py`)
  - other DragDiffusion utilities (drag update, attention control, LoRA key fixing, etc.)
- `drag_pipeline.py` – DragDiffusion pipeline (built on top of `diffusers`)

---

### 2. Environment and Dependencies

We follow the original DragDiffusion environment as closely as possible.

- **OS**: Linux
- **GPU**: NVIDIA GPU with ≥ 14 GB VRAM
- **Python**: 3.9+

Create and activate the Conda environment (from the original repo):

```bash
conda env create -f environment.yaml
conda activate dragdiff
```

Key libraries (non-exhaustive):

- `torch`, `torchvision`
- `diffusers>=0.24.0`
- `transformers`
- `accelerate`
- `lpips`
- `einops`
- `pytorch-lightning`

---

### 3. Data: DragBench

We use the **DragBench** dataset released with DragDiffusion.

Expected layout (inside `drag_bench_evaluation/drag_bench_data/`):

- 10 categories:
  - `art_work/`, `land_scape/`, `building_city_view/`, `building_countryside_view/`,
    `animals/`, `human_head/`, `human_upper_body/`, `human_full_body/`,
    `interior_design/`, `other_objects/`
- For each sample:
  - `original_image.png`
  - `meta_data.pkl` containing:
    - `prompt` (text prompt)
    - `mask` (editable region)
    - `points` (handle/target point pairs)

Place the dataset under:

```text
drag_bench_evaluation/drag_bench_data/
```

following the instructions from the original DragDiffusion `README`.

---

### 4. LoRA Training on DragBench

**Goal:** Train one LoRA per DragBench image so DragDiffusion can specialize to each input.

We use `drag_bench_evaluation/run_lora_training.py`, with:

- **LoRA training steps**: up to **120** steps per image
- Checkpoints saved every 10 steps:
  - `10/`, `20/`, …, `120/`, each containing `pytorch_lora_weights.safetensors`

Run:

```bash
cd drag_bench_evaluation
python run_lora_training.py
```

This:

- Iterates over all categories and samples in `drag_bench_data/`
- Trains LoRA for each sample up to 120 steps with:
  - `model_path = "runwayml/stable-diffusion-v1-5"`
  - `lora_step = 120`
  - `lora_lr = 5e-4`
  - `lora_batch_size = 4`
  - `lora_rank = 16`
  - `save_interval = 10`

Resulting LoRA structure example:

```text
drag_bench_lora/animals/JH_2023-09-14-1820-16/
  10/pytorch_lora_weights.safetensors
  20/pytorch_lora_weights.safetensors
  ...
  120/pytorch_lora_weights.safetensors
  pytorch_lora_weights.safetensors    # final full LoRA
```

---

### 5. DragDiffusion Evaluation on DragBench

#### 5.1. Core metrics

We rely on two core metrics:

- **Mean Distance (MD)** – implemented in `run_eval_point_matching.py`
  - Uses DIFT features (`dift_sd.py`) based on SD-2.1
  - For each handle/target point pair:
    - Finds the best-matching location in the dragged image
    - Computes Euclidean distance between this match and the target point
  - Reports the average distance across all points and samples.

- **Image Fidelity (IF)** – implemented in `run_eval_image_fidelity.py`
  - Uses LPIPS (AlexNet backbone)
  - Computes LPIPS between original and edited image
  - Defines IF as:
    \[
      \text{IF} = 1 - \text{LPIPS}
    \]
  - Reports mean IF across all samples.

Our `run_timestep_ablation.py` script wraps both metrics and evaluates them across all categories for each configuration.

---

### 6. Timestep and LoRA Ablation Script

We added a unified ablation script:

- **File**: `drag_bench_evaluation/run_timestep_ablation.py`
- **Purpose**:
  - Run DragDiffusion on the full DragBench set
  - Vary:
    - Optimized timestep \(t\) (via `inv_strength`)
    - LoRA training steps (including **no LoRA**)
  - Evaluate **MD** and **IF** for each configuration
  - Print a concise summary table

#### 6.1. Interface

```bash
cd drag_bench_evaluation

# Example: LoRA steps ablation at t = 35
python run_timestep_ablation.py \
  --run --eval \
  --timesteps 35 \
  --lora_steps 0 20 40 80 100 120
```

Arguments:

- `--timesteps`:
  - Timestep \(t\) values in DDIM space (for **DDIM steps = 50**).
  - Internally mapped via:
    \[
      t = \text{round}(\text{inv\_strength} \times 50)
    \]
  - Implemented mapping:
    - 10 → inv_strength=0.2  
    - 20 → 0.4  
    - 30 → 0.6  
    - 35 → 0.7  
    - 40 → 0.8  
    - 50 → 1.0

- `--lora_steps`:
  - LoRA training steps to evaluate (e.g. `0 20 40 80 100 120`).
  - `0` means **no LoRA** (we skip LoRA loading in `run_drag`).

- `--run`:
  - Runs DragDiffusion and writes results to:
    ```text
    drag_diffusion_res_{lora_steps}_{inv_strength}_0.01_3/
    ```

- `--eval`:
  - Runs:
    - `evaluate_md` (MD)
    - `evaluate_if` (IF)
  - Prints per-setting metrics and a summary table.

#### 6.2. Example experiments

- **Timestep ablation** (LoRA steps fixed at 80):

  ```bash
  python run_timestep_ablation.py \
    --run --eval \
    --timesteps 20 35 50 \
    --lora_steps 80
  ```

- **LoRA steps ablation** (t fixed at 35, inv_strength=0.7):

  ```bash
  python run_timestep_ablation.py \
    --run --eval \
    --timesteps 35 \
    --lora_steps 0 20 40 80 100 120
  ```

Each run prints a summary table of the form:

```text
============================================================
SUMMARY: Ablation Results
============================================================
     t |   LoRA Steps |         MD | IF (1-LPIPS)
--------------------------------------------------
    35 |            0 |     ...    |      ...
    35 |           20 |     ...    |      ...
    35 |           40 |     ...    |      ...
    35 |           80 |     ...    |      ...
    35 |          100 |     ...    |      ...
    35 |          120 |     ...    |      ...
============================================================
```

---

### 7. Other Ablations (from original repo)

The repo also includes the original ablation scripts:

- **Lambda (regularization) ablation**:
  - `eval_lambda_ablation.sh`
  - Result dirs like:
    - `drag_diffusion_res_80_0.7_0.01_3_lam0.0`
    - `drag_diffusion_res_80_0.7_0.01_3_lam0.1`
    - `drag_diffusion_res_80_0.7_0.01_3_lam0.5`
    - `drag_diffusion_res_80_0.7_0.01_3_lam1.0`

- **UNet block ablation**:
  - `run_unet_block_ablation.sh`
  - Result dirs like:
    - `drag_diffusion_res_80_0.7_0.01_1_lam0.1`
    - `drag_diffusion_res_80_0.7_0.01_2_lam0.1`
    - `drag_diffusion_res_80_0.7_0.01_3_lam0.1`
    - `drag_diffusion_res_80_0.7_0.01_4_lam0.1`

These help isolate the effect of regularization strength \(\lambda\) and the UNet supervision layer.

---

### 8. How to Reproduce Our Reported Results

1. **Set up environment** as in Section 2.
2. **Download DragBench** and place it under `drag_bench_evaluation/drag_bench_data/`.
3. **Train LoRA for all images**:

   ```bash
   cd drag_bench_evaluation
   python run_lora_training.py
   ```

4. **Run desired ablations**, for example:

   - LoRA steps ablation (t = 35):

     ```bash
     python run_timestep_ablation.py \
       --run --eval \
       --timesteps 35 \
       --lora_steps 0 20 40 80 100 120
     ```

   - Timestep ablation (LoRA steps = 80):

     ```bash
     python run_timestep_ablation.py \
       --run --eval \
       --timesteps 20 35 50 \
       --lora_steps 80
     ```

5. **Collect metrics** from the printed summary tables (MD and IF).

---

### 9. Citation

If you use this code or our analysis, please cite the original DragDiffusion paper and the TMLR reproducibility paper (once available):

```bibtex
@article{shi2023dragdiffusion,
  title={DragDiffusion: Harnessing Diffusion Models for Interactive Point-based Image Editing},
  author={Shi, Yujun and Xue, Chuhui and Pan, Jiachun and Zhang, Wenqing and Tan, Vincent YF and Bai, Song},
  journal={arXiv preprint arXiv:2306.14435},
  year={2023}
}
```

