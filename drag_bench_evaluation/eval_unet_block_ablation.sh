#!/bin/bash
# Run LPIPS and mean distance evaluations for UNet decoder block ablation experiments
# UNet decoder block indices: 1, 2, 3, 4
# Settings: DDIM=50, t=35 (inv_strength=0.7), LoRA=80 steps, lambda=0.1

cd /shared/home/ali.subhan/DragDiffusion/drag_bench_evaluation

# UNet decoder block indices tested
unet_feature_indices=(1 2 3 4)

# Fixed settings
LORA_STEPS=80
INV_STRENGTH=0.7
LATENT_LR=0.01
LAMBDA_REG=0.1

echo "=========================================="
echo "Evaluating UNet Decoder Block Ablation Experiments"
echo "=========================================="
echo "Settings:"
echo "  - LoRA steps: $LORA_STEPS"
echo "  - Inversion strength: $INV_STRENGTH (t=35)"
echo "  - Latent LR: $LATENT_LR"
echo "  - Lambda: $LAMBDA_REG"
echo "  - UNet decoder block indices: ${unet_feature_indices[@]}"
echo ""

# Build list of result directories
eval_roots=()
for unet_idx in "${unet_feature_indices[@]}"; do
    result_dir="drag_diffusion_res_${LORA_STEPS}_${INV_STRENGTH}_${LATENT_LR}_${unet_idx}_lam${LAMBDA_REG}"
    if [ -d "$result_dir" ]; then
        eval_roots+=("$result_dir")
        echo "Found result directory: $result_dir"
    else
        echo "WARNING: Result directory not found: $result_dir"
    fi
done

if [ ${#eval_roots[@]} -eq 0 ]; then
    echo "ERROR: No result directories found. Please run the experiments first."
    exit 1
fi

echo ""
echo "=========================================="
echo "1. Running LPIPS/CLIP Similarity Evaluation"
echo "=========================================="
python3 run_eval_similarity.py --eval_root "${eval_roots[@]}"

echo ""
echo "=========================================="
echo "2. Running Mean Distance Evaluation"
echo "=========================================="
python3 run_eval_point_matching.py --eval_root "${eval_roots[@]}"

echo ""
echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="

