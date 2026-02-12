# Sampled Results - Image Copy Summary

## Overview
This document describes the organization of the `sampled_results` folder, which contains copied and renamed images from various drag diffusion evaluation experiments.

## Script
The images were copied using: `copy_sample_images.py`

## Directory Structure

```
sampled_results/
├── animals/
├── art_work/
├── building_city_view/
├── building_countryside_view/
├── human_full_body/
├── human_head/
├── human_upper_body/
├── interior_design/
├── land_scape/
└── other_objects/
```

## Naming Convention

Each image follows the naming pattern:
```
{sample_number}_{category}_{experiment_name}.png
```

**Examples:**
- `1_animals_drag_diffusion_res_80_0.7_0.01_1_lam0.1.png`
- `2_human_head_drag_diffusion_res_0_0.7_0.01_3.png`
- `3_art_work_multi_timestep_baseline.png`

**Components:**
- `sample_number`: 1, 2, or 3 (first 3 samples from each category)
- `category`: The category name (e.g., animals, human_head, etc.)
- `experiment_name`: The root directory name indicating the experiment configuration

## Statistics

Total images per category: **54 images**

This includes 3 samples from each of the following 18 experiment configurations:
1. `drag_diffusion_res_0_0.7_0.01_3`
2. `drag_diffusion_res_20_0.7_0.01_3`
3. `drag_diffusion_res_40_0.7_0.01_3`
4. `drag_diffusion_res_80_0.4_0.01_3`
5. `drag_diffusion_res_80_0.7_0.01_1_lam0.1`
6. `drag_diffusion_res_80_0.7_0.01_2_lam0.1`
7. `drag_diffusion_res_80_0.7_0.01_3`
8. `drag_diffusion_res_80_0.7_0.01_3_lam0.0`
9. `drag_diffusion_res_80_0.7_0.01_3_lam0.1`
10. `drag_diffusion_res_80_0.7_0.01_3_lam0.5`
11. `drag_diffusion_res_80_0.7_0.01_3_lam1.0`
12. `drag_diffusion_res_80_0.7_0.01_4_lam0.1`
13. `drag_diffusion_res_80_1.0_0.01_3`
14. `drag_diffusion_res_100_0.7_0.01_3`
15. `drag_diffusion_res_120_0.7_0.01_3`
16. `drag_diffusion_res_no_lora_0.7_0.01_3`
17. `multi_timestep_baseline`
18. `multi_timestep_multi`

**Note:** `drag_diffusion_res_200_0.7_0.01_3` was skipped as it contained no sample images.

## Categories

All 10 categories are represented:
1. **animals** - 54 images
2. **art_work** - 54 images
3. **building_city_view** - 54 images
4. **building_countryside_view** - 54 images
5. **human_full_body** - 54 images
6. **human_head** - 54 images
7. **human_upper_body** - 54 images
8. **interior_design** - 54 images
9. **land_scape** - 54 images
10. **other_objects** - 54 images

**Total images:** 540 images

## Source Structure

Images were copied from:
```
drag_bench_evaluation/
└── {experiment_name}/
    └── {category}/
        └── {sample_folder}/
            └── dragged_image.png
```

The first 3 sample folders (sorted alphabetically) from each category were selected.
