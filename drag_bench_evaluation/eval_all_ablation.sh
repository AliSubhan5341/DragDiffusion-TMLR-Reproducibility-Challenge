#!/bin/bash
# Run LPIPS and mean distance evaluations for ALL ablation experiments
# Includes both lambda ablation and UNet decoder block ablation

cd /shared/home/ali.subhan/DragDiffusion/drag_bench_evaluation

echo "=========================================="
echo "Evaluating ALL Ablation Experiments"
echo "=========================================="
echo ""

# Lambda ablation evaluation
echo "Running Lambda Ablation Evaluation..."
bash eval_lambda_ablation.sh

echo ""
echo "=========================================="
echo ""

# UNet block ablation evaluation
echo "Running UNet Decoder Block Ablation Evaluation..."
bash eval_unet_block_ablation.sh

echo ""
echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="

