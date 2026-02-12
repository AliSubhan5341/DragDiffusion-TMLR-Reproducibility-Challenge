#!/usr/bin/env python
"""
Run DragDiffusion timestep ablation study on DragBench.
Varies optimized timestep t = 10, 20, 30, 35, 40, 50 with DDIM=50 fixed and LoRA enabled.
Reports Mean Distance (MD) and Image Fidelity (IF = 1 - LPIPS) for each.
"""

import os
import sys
import pickle
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import PILToTensor
import lpips
from einops import rearrange
from pytorch_lightning import seed_everything
from copy import deepcopy
from types import SimpleNamespace

from diffusers import DDIMScheduler, AutoencoderKL
from torchvision.utils import save_image

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drag_pipeline import DragPipeline
from utils.drag_utils import drag_diffusion_update
from utils.attn_utils import register_attention_editor_diffusers, MutualSelfAttentionControl
from utils.fix_lora import fix_lora_keys

# Import DIFT for point matching evaluation
from dift_sd import SDFeaturizer

ALL_CATEGORIES = [
    'art_work', 'land_scape', 'building_city_view', 'building_countryside_view',
    'animals', 'human_head', 'human_upper_body', 'human_full_body',
    'interior_design', 'other_objects',
]

# Timestep to inv_strength mapping (t = round(inv_strength * 50))
TIMESTEP_TO_INV = {
    10: 0.2,
    20: 0.4,
    30: 0.6,
    35: 0.7,
    40: 0.8,
    50: 1.0,
}


def preprocess_image(image, device):
    image = torch.from_numpy(image).float() / 127.5 - 1
    image = rearrange(image, "h w c -> 1 c h w")
    image = image.to(device)
    return image


def run_drag(source_image, mask, prompt, points, inversion_strength,
             lam, latent_lr, unet_feature_idx, n_pix_step,
             model_path, vae_path, lora_path, start_step, start_layer):
    """Run drag diffusion on a single image."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                              beta_schedule="scaled_linear", clip_sample=False,
                              set_alpha_to_one=False, steps_offset=1)
    model = DragPipeline.from_pretrained(model_path, scheduler=scheduler).to(device)
    model.modify_unet_forward()

    if vae_path != "default":
        model.vae = AutoencoderKL.from_pretrained(vae_path).to(model.vae.device, model.vae.dtype)

    seed = 42
    seed_everything(seed)

    args = SimpleNamespace()
    args.prompt = prompt
    args.points = points
    args.n_inference_step = 50
    args.n_actual_inference_step = round(inversion_strength * args.n_inference_step)
    args.guidance_scale = 1.0
    args.unet_feature_idx = [unet_feature_idx]
    args.r_m = 1
    args.r_p = 3
    args.lam = lam
    args.lr = latent_lr
    args.n_pix_step = n_pix_step

    full_h, full_w = source_image.shape[:2]
    args.sup_res_h = int(0.5 * full_h)
    args.sup_res_w = int(0.5 * full_w)

    source_image_tensor = preprocess_image(source_image, device)

    if lora_path == "":
        model.unet.set_default_attn_processor()
    else:
        fixed_lora_file = os.path.join(lora_path, "pytorch_lora_weights_fixed.safetensors")
        if not os.path.exists(fixed_lora_file):
            fix_lora_keys(lora_path)
        model.load_lora_weights(lora_path, weight_name="pytorch_lora_weights_fixed.safetensors")

    invert_code = model.invert(source_image_tensor, prompt,
                               guidance_scale=args.guidance_scale,
                               num_inference_steps=args.n_inference_step,
                               num_actual_inference_steps=args.n_actual_inference_step)

    mask_tensor = torch.from_numpy(mask).float() / 255.
    mask_tensor[mask_tensor > 0.0] = 1.0
    mask_tensor = rearrange(mask_tensor, "h w -> 1 1 h w").cuda()
    mask_tensor = F.interpolate(mask_tensor, (args.sup_res_h, args.sup_res_w), mode="nearest")

    handle_points = []
    target_points = []
    for idx, point in enumerate(points):
        cur_point = torch.tensor([point[1] / full_h * args.sup_res_h, point[0] / full_w * args.sup_res_w])
        cur_point = torch.round(cur_point)
        if idx % 2 == 0:
            handle_points.append(cur_point)
        else:
            target_points.append(cur_point)

    init_code = invert_code
    init_code_orig = deepcopy(init_code)
    model.scheduler.set_timesteps(args.n_inference_step)
    t = model.scheduler.timesteps[args.n_inference_step - args.n_actual_inference_step]

    updated_init_code = drag_diffusion_update(model, init_code, None, t, handle_points, target_points, mask_tensor, args)

    editor = MutualSelfAttentionControl(start_step=start_step, start_layer=start_layer,
                                        total_steps=args.n_inference_step, guidance_scale=args.guidance_scale)
    if lora_path == "":
        register_attention_editor_diffusers(model, editor, attn_processor='attn_proc')
    else:
        register_attention_editor_diffusers(model, editor, attn_processor='lora_attn_proc')

    gen_image = model(prompt=args.prompt, batch_size=2,
                      latents=torch.cat([init_code_orig, updated_init_code], dim=0),
                      guidance_scale=args.guidance_scale,
                      num_inference_steps=args.n_inference_step,
                      num_actual_inference_steps=args.n_actual_inference_step)[1].unsqueeze(dim=0)

    gen_image = F.interpolate(gen_image, (full_h, full_w), mode='bilinear')
    out_image = gen_image.cpu().permute(0, 2, 3, 1).numpy()[0]
    out_image = (out_image * 255).astype(np.uint8)
    return out_image


def run_drag_diffusion_for_inv(inv_strength, lora_steps, base_dir):
    """Run DragDiffusion on all DragBench samples with given inversion strength."""
    root_dir = os.path.join(base_dir, 'drag_bench_data')
    lora_dir = os.path.join(base_dir, 'drag_bench_lora')
    result_dir = os.path.join(base_dir, f'drag_diffusion_res_{lora_steps}_{inv_strength}_0.01_3')

    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
        for cat in ALL_CATEGORIES:
            os.makedirs(os.path.join(result_dir, cat), exist_ok=True)

    for cat in ALL_CATEGORIES:
        file_dir = os.path.join(root_dir, cat)
        if not os.path.exists(file_dir):
            print(f"Warning: {file_dir} not found, skipping")
            continue
        for sample_name in os.listdir(file_dir):
            if sample_name == '.DS_Store':
                continue
            sample_path = os.path.join(file_dir, sample_name)

            # Check if already processed
            save_dir = os.path.join(result_dir, cat, sample_name)
            if os.path.exists(os.path.join(save_dir, 'dragged_image.png')):
                print(f"Skipping {cat}/{sample_name} (already exists)")
                continue

            source_image = Image.open(os.path.join(sample_path, 'original_image.png'))
            source_image = np.array(source_image)

            with open(os.path.join(sample_path, 'meta_data.pkl'), 'rb') as f:
                meta_data = pickle.load(f)
            prompt = meta_data['prompt']
            mask = meta_data['mask']
            points = meta_data['points']

            if lora_steps <= 0:
                lora_path = ""
            else:
                lora_path = os.path.join(lora_dir, cat, sample_name, str(lora_steps))
            print(f"Processing {cat}/{sample_name} with inv_strength={inv_strength}, lora_steps={lora_steps}")

            out_image = run_drag(
                source_image, mask, prompt, points,
                inversion_strength=inv_strength,
                lam=0.1, latent_lr=0.01, unet_feature_idx=3, n_pix_step=80,
                model_path="runwayml/stable-diffusion-v1-5",
                vae_path="default", lora_path=lora_path,
                start_step=0, start_layer=10,
            )

            if not os.path.isdir(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            Image.fromarray(out_image).save(os.path.join(save_dir, 'dragged_image.png'))


def evaluate_md(result_dir, original_dir):
    """Evaluate Mean Distance using DIFT features."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dift = SDFeaturizer('Manojb/stable-diffusion-2-1-base')
    seed_everything(42)

    all_dist = []
    for cat in ALL_CATEGORIES:
        cat_path = os.path.join(original_dir, cat)
        if not os.path.exists(cat_path):
            continue
        for file_name in os.listdir(cat_path):
            if file_name == '.DS_Store':
                continue

            meta_path = os.path.join(original_dir, cat, file_name, 'meta_data.pkl')
            if not os.path.exists(meta_path):
                continue

            with open(meta_path, 'rb') as f:
                meta_data = pickle.load(f)
            prompt = meta_data['prompt']
            points = meta_data['points']

            handle_points, target_points = [], []
            for idx, point in enumerate(points):
                cur_point = torch.tensor([point[1], point[0]])
                if idx % 2 == 0:
                    handle_points.append(cur_point)
                else:
                    target_points.append(cur_point)

            source_path = os.path.join(original_dir, cat, file_name, 'original_image.png')
            dragged_path = os.path.join(result_dir, cat, file_name, 'dragged_image.png')

            if not os.path.exists(dragged_path):
                print(f"Warning: {dragged_path} not found, skipping")
                continue

            source_img = Image.open(source_path)
            dragged_img = Image.open(dragged_path).resize(source_img.size, Image.BILINEAR)

            source_tensor = (PILToTensor()(source_img) / 255.0 - 0.5) * 2
            dragged_tensor = (PILToTensor()(dragged_img) / 255.0 - 0.5) * 2
            _, H, W = source_tensor.shape

            ft_source = dift.forward(source_tensor, prompt=prompt, t=261, up_ft_index=1, ensemble_size=8)
            ft_source = F.interpolate(ft_source, (H, W), mode='bilinear')

            ft_dragged = dift.forward(dragged_tensor, prompt=prompt, t=261, up_ft_index=1, ensemble_size=8)
            ft_dragged = F.interpolate(ft_dragged, (H, W), mode='bilinear')

            cos = nn.CosineSimilarity(dim=1)
            for pt_idx in range(len(handle_points)):
                hp = handle_points[pt_idx]
                tp = target_points[pt_idx]
                num_channel = ft_source.size(1)
                src_vec = ft_source[0, :, hp[0], hp[1]].view(1, num_channel, 1, 1)
                cos_map = cos(src_vec, ft_dragged).cpu().numpy()[0]
                max_rc = np.unravel_index(cos_map.argmax(), cos_map.shape)
                dist = (tp - torch.tensor(max_rc)).float().norm()
                all_dist.append(dist.item())

    return np.mean(all_dist) if all_dist else float('nan')


def evaluate_if(result_dir, original_dir):
    """Evaluate Image Fidelity (1 - LPIPS)."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    loss_fn = lpips.LPIPS(net="alex").to(device)

    all_if = []
    for cat in ALL_CATEGORIES:
        cat_path = os.path.join(original_dir, cat)
        if not os.path.exists(cat_path):
            continue
        for file_name in os.listdir(cat_path):
            if file_name == '.DS_Store':
                continue

            source_path = os.path.join(original_dir, cat, file_name, 'original_image.png')
            dragged_path = os.path.join(result_dir, cat, file_name, 'dragged_image.png')

            if not os.path.exists(dragged_path):
                continue

            source_img = np.array(Image.open(source_path).convert('RGB'))
            dragged_img = Image.open(dragged_path).convert('RGB')
            dragged_img = dragged_img.resize((source_img.shape[1], source_img.shape[0]), Image.BILINEAR)
            dragged_img = np.array(dragged_img)

            source_tensor = torch.from_numpy(source_img).float() / 127.5 - 1.0
            source_tensor = rearrange(source_tensor, "h w c -> 1 c h w").to(device)
            dragged_tensor = torch.from_numpy(dragged_img).float() / 127.5 - 1.0
            dragged_tensor = rearrange(dragged_tensor, "h w c -> 1 c h w").to(device)

            with torch.no_grad():
                lpips_val = loss_fn(source_tensor, dragged_tensor).item()
            all_if.append(1.0 - lpips_val)

    return np.mean(all_if) if all_if else float('nan')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true', help='Run DragDiffusion experiments')
    parser.add_argument('--eval', action='store_true', help='Evaluate existing results')
    parser.add_argument('--timesteps', type=int, nargs='+', default=[10, 20, 30, 35, 40, 50])
    parser.add_argument('--lora_steps', type=int, nargs='+', default=[80])
    args = parser.parse_args()

    # Use absolute path based on script location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    original_dir = os.path.join(base_dir, 'drag_bench_data')

    results = {}

    for t in args.timesteps:
        inv_strength = TIMESTEP_TO_INV[t]
        for lora_steps in args.lora_steps:
            result_dir = os.path.join(base_dir, f'drag_diffusion_res_{lora_steps}_{inv_strength}_0.01_3')

            if args.run:
                print(f"\n{'='*60}")
                print(f"Running DragDiffusion with t={t} (inv_strength={inv_strength}), lora_steps={lora_steps}")
                print(f"{'='*60}")
                run_drag_diffusion_for_inv(inv_strength, lora_steps, base_dir)

            if args.eval or not args.run:
                print(f"\nEvaluating t={t} (inv_strength={inv_strength}), lora_steps={lora_steps}...")
                md = evaluate_md(result_dir, original_dir)
                if_score = evaluate_if(result_dir, original_dir)
                key = (t, lora_steps)
                results[key] = {'MD': md, 'IF': if_score}
                print(f"  t={t}, lora_steps={lora_steps}: MD={md:.4f}, IF={if_score:.4f}")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY: Ablation Results")
    print("=" * 60)
    print(f"{'t':>6} | {'LoRA Steps':>12} | {'MD':>10} | {'IF (1-LPIPS)':>12}")
    print("-" * 50)
    for key in sorted(results.keys()):
        t, lora_steps = key
        print(f"{t:>6} | {lora_steps:>12} | {results[key]['MD']:>10.4f} | {results[key]['IF']:>12.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
