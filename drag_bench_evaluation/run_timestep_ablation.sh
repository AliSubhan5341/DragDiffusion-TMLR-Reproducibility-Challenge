#!/bin/bash
# Run DragDiffusion on DragBench with LoRA, DDIM=50, varying timestep t

# t = n_actual_inference_step = round(inv_strength * 50)
# t=10 -> inv_strength=0.2
# t=20 -> inv_strength=0.4
# t=30 -> inv_strength=0.6
# t=35 -> inv_strength=0.7
# t=40 -> inv_strength=0.8
# t=50 -> inv_strength=1.0

cd /shared/home/ali.subhan/DragDiffusion/drag_bench_evaluation

for inv in 0.2 0.4 0.6 0.7 0.8 1.0; do
    echo "Running with inv_strength=$inv"
    python run_drag_diffusion.py --inv_strength $inv --latent_lr 0.01 --unet_feature_idx 3 --lora_steps 80
done

echo "All runs complete. Now evaluating..."

# Evaluate MD (Mean Distance)
for inv in 0.2 0.4 0.6 0.7 0.8 1.0; do
    result_dir="drag_diffusion_res_80_${inv}_0.01_3"
    echo "Evaluating MD for $result_dir"
    python run_eval_point_matching.py --eval_root $result_dir
done




