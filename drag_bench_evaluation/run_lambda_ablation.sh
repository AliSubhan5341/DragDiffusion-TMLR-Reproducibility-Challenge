#!/bin/bash
# Run drag diffusion experiments with different lambda (mask regularization) values
# Settings: DDIM=50, t=35 (inv_strength=0.7), LoRA=80 steps, n_pix_step=120

cd /shared/home/ali.subhan/DragDiffusion/drag_bench_evaluation

# Lambda values to test
lambda_values=(0.0 0.1 0.5 1.0)

# Fixed settings
LORA_STEPS=80
INV_STRENGTH=0.7  # This gives t=35 with DDIM=50 (35/50 = 0.7)
LATENT_LR=0.01
UNET_FEATURE_IDX=3

echo "Running lambda ablation study..."
echo "Settings:"
echo "  - DDIM steps: 50"
echo "  - Inversion strength: $INV_STRENGTH (t=35)"
echo "  - LoRA steps: $LORA_STEPS"
echo "  - Latent LR: $LATENT_LR"
echo "  - UNet feature idx: $UNET_FEATURE_IDX"
echo "  - Lambda values: ${lambda_values[@]}"
echo ""

for lambda in "${lambda_values[@]}"; do
    echo "=========================================="
    echo "Running with lambda = $lambda"
    echo "=========================================="
    python3 run_drag_diffusion.py \
        --lora_steps $LORA_STEPS \
        --inv_strength $INV_STRENGTH \
        --latent_lr $LATENT_LR \
        --unet_feature_idx $UNET_FEATURE_IDX \
        --lambda_reg $lambda
    echo ""
done

echo "All lambda ablation experiments completed!"

