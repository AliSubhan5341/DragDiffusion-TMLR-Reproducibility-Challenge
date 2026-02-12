#!/bin/bash
# Run drag diffusion experiments with different UNet decoder block feature maps
# Settings: DDIM=50, t=35 (inv_strength=0.7), LoRA=80 steps, lambda=0.1

cd /shared/home/ali.subhan/DragDiffusion/drag_bench_evaluation

# UNet decoder block indices to test (1, 2, 3, 4)
# Note: Index 0 is mid block, indices 1-4 are decoder blocks 1-4
unet_feature_indices=(1 2 3 4)

# Fixed settings
LORA_STEPS=80
INV_STRENGTH=0.7  # This gives t=35 with DDIM=50 (35/50 = 0.7)
LATENT_LR=0.01
LAMBDA_REG=0.1

echo "Running UNet decoder block ablation study..."
echo "Settings:"
echo "  - DDIM steps: 50"
echo "  - Inversion strength: $INV_STRENGTH (t=35)"
echo "  - LoRA steps: $LORA_STEPS"
echo "  - Latent LR: $LATENT_LR"
echo "  - Lambda: $LAMBDA_REG"
echo "  - UNet decoder block indices: ${unet_feature_indices[@]}"
echo "  (Index 0 = mid block, 1-4 = decoder blocks 1-4)"
echo ""

for unet_idx in "${unet_feature_indices[@]}"; do
    echo "=========================================="
    echo "Running with UNet decoder block $unet_idx (feature index $unet_idx)"
    echo "=========================================="
    python3 run_drag_diffusion.py \
        --lora_steps $LORA_STEPS \
        --inv_strength $INV_STRENGTH \
        --latent_lr $LATENT_LR \
        --unet_feature_idx $unet_idx \
        --lambda_reg $LAMBDA_REG
    echo ""
done

echo "All UNet decoder block ablation experiments completed!"

