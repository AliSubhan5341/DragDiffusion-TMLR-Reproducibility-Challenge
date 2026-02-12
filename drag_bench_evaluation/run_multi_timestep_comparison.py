# *************************************************************************
# Copyright (2023) Bytedance Inc.
#
# Copyright (2023) DragDiffusion Authors 
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 
#
#     http://www.apache.org/licenses/LICENSE-2.0 
#
# Unless required by applicable law or agreed to in writing, software 
# distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
# *************************************************************************

# Compare baseline single-timestep (t=35) vs multi-timestep (t_set={30,35,40})
import argparse
import os
import time
import datetime
import numpy as np
import torch
import torch.nn.functional as F
import pickle
import PIL
from PIL import Image
from copy import deepcopy
from einops import rearrange
from types import SimpleNamespace

from diffusers import DDIMScheduler, AutoencoderKL
from pytorch_lightning import seed_everything

import sys
sys.path.insert(0, '../')
from drag_pipeline import DragPipeline

from utils.drag_utils import drag_diffusion_update, drag_diffusion_update_multi_timestep
from utils.attn_utils import register_attention_editor_diffusers, MutualSelfAttentionControl
from utils.fix_lora import fix_lora_keys

# Import evaluation functions
import lpips
import clip
from torchvision.transforms import PILToTensor
from dift_sd import SDFeaturizer


def preprocess_image(image, device):
    image = torch.from_numpy(image).float() / 127.5 - 1 # [-1, 1]
    image = rearrange(image, "h w c -> 1 c h w")
    image = image.to(device)
    return image


def run_drag_single_timestep(source_image, mask, prompt, points, t, inversion_strength,
                             lam, latent_lr, unet_feature_idx, n_pix_step,
                             model_path, vae_path, lora_path, start_step, start_layer):
    """Baseline: single timestep optimization at t"""
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
    args.sup_res_h = int(0.5*full_h)
    args.sup_res_w = int(0.5*full_w)

    source_image_tensor = preprocess_image(source_image, device)

    # set lora
    if lora_path == "":
        model.unet.set_default_attn_processor()
    else:
        fixed_lora_file = os.path.join(lora_path, "pytorch_lora_weights_fixed.safetensors")
        if not os.path.exists(fixed_lora_file):
            fix_lora_keys(lora_path)
        model.load_lora_weights(lora_path, weight_name="pytorch_lora_weights_fixed.safetensors")

    # invert the source image
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
        cur_point = torch.tensor([point[1]/full_h*args.sup_res_h, point[0]/full_w*args.sup_res_w])
        cur_point = torch.round(cur_point)
        if idx % 2 == 0:
            handle_points.append(cur_point)
        else:
            target_points.append(cur_point)

    init_code = invert_code
    model.scheduler.set_timesteps(args.n_inference_step)
    t_tensor = model.scheduler.timesteps[args.n_inference_step - args.n_actual_inference_step]

    # Run optimization
    start_time = time.time()
    updated_init_code = drag_diffusion_update(model, init_code, None, t_tensor,
        handle_points, target_points, mask_tensor, args)
    runtime = time.time() - start_time

    # Inference
    editor = MutualSelfAttentionControl(start_step=start_step, start_layer=start_layer,
                                        total_steps=args.n_inference_step,
                                        guidance_scale=args.guidance_scale)
    if lora_path == "":
        register_attention_editor_diffusers(model, editor, attn_processor='attn_proc')
    else:
        register_attention_editor_diffusers(model, editor, attn_processor='lora_attn_proc')

    init_code_orig = deepcopy(invert_code)
    gen_image = model(prompt=args.prompt, batch_size=2,
        latents=torch.cat([init_code_orig, updated_init_code], dim=0),
        guidance_scale=args.guidance_scale, num_inference_steps=args.n_inference_step,
        num_actual_inference_steps=args.n_actual_inference_step)[1].unsqueeze(dim=0)

    gen_image = F.interpolate(gen_image, (full_h, full_w), mode='bilinear')
    out_image = gen_image.cpu().permute(0, 2, 3, 1).numpy()[0]
    out_image = (out_image * 255).astype(np.uint8)
    
    return out_image, runtime


def run_drag_multi_timestep(source_image, mask, prompt, points, t_set, inversion_strength,
                            lam, latent_lr, unet_feature_idx, n_pix_step,
                            model_path, vae_path, lora_path, start_step, start_layer):
    """Multi-timestep: optimize at multiple timesteps simultaneously"""
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
    args.sup_res_h = int(0.5*full_h)
    args.sup_res_w = int(0.5*full_w)

    source_image_tensor = preprocess_image(source_image, device)

    # set lora
    if lora_path == "":
        model.unet.set_default_attn_processor()
    else:
        fixed_lora_file = os.path.join(lora_path, "pytorch_lora_weights_fixed.safetensors")
        if not os.path.exists(fixed_lora_file):
            fix_lora_keys(lora_path)
        model.load_lora_weights(lora_path, weight_name="pytorch_lora_weights_fixed.safetensors")

    # invert the source image
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
        cur_point = torch.tensor([point[1]/full_h*args.sup_res_h, point[0]/full_w*args.sup_res_w])
        cur_point = torch.round(cur_point)
        if idx % 2 == 0:
            handle_points.append(cur_point)
        else:
            target_points.append(cur_point)

    init_code = invert_code
    model.scheduler.set_timesteps(args.n_inference_step)
    
    # Convert t_set step indices to actual timestep tensors
    # t_set contains step indices (e.g., 30, 35, 40) which correspond to:
    # step 30 -> timestep index 50-30=20, step 35 -> index 50-35=15, step 40 -> index 50-40=10
    t_tensors = []
    for t_step in t_set:
        # t_step is the DDIM step number (0-49), convert to timestep array index
        timestep_idx = args.n_inference_step - t_step
        if timestep_idx < 0 or timestep_idx >= len(model.scheduler.timesteps):
            raise ValueError(f"Invalid timestep step {t_step} for {args.n_inference_step} DDIM steps")
        t_tensor = model.scheduler.timesteps[timestep_idx]
        t_tensors.append(t_tensor)

    # Run multi-timestep optimization
    start_time = time.time()
    updated_init_code = drag_diffusion_update_multi_timestep(model, init_code, None, t_tensors,
        handle_points, target_points, mask_tensor, args)
    runtime = time.time() - start_time

    # Inference (use middle timestep for inference)
    t_inference = t_tensors[len(t_tensors)//2]
    editor = MutualSelfAttentionControl(start_step=start_step, start_layer=start_layer,
                                        total_steps=args.n_inference_step,
                                        guidance_scale=args.guidance_scale)
    if lora_path == "":
        register_attention_editor_diffusers(model, editor, attn_processor='attn_proc')
    else:
        register_attention_editor_diffusers(model, editor, attn_processor='lora_attn_proc')

    init_code_orig = deepcopy(invert_code)
    gen_image = model(prompt=args.prompt, batch_size=2,
        latents=torch.cat([init_code_orig, updated_init_code], dim=0),
        guidance_scale=args.guidance_scale, num_inference_steps=args.n_inference_step,
        num_actual_inference_steps=args.n_actual_inference_step)[1].unsqueeze(dim=0)

    gen_image = F.interpolate(gen_image, (full_h, full_w), mode='bilinear')
    out_image = gen_image.cpu().permute(0, 2, 3, 1).numpy()[0]
    out_image = (out_image * 255).astype(np.uint8)
    
    return out_image, runtime


def compute_lpips(source_image, dragged_image, loss_fn_alex, device):
    """Compute LPIPS score"""
    source_tensor = preprocess_image(source_image, device)
    dragged_tensor = preprocess_image(dragged_image, device)
    
    with torch.no_grad():
        source_224 = F.interpolate(source_tensor, (224, 224), mode='bilinear')
        dragged_224 = F.interpolate(dragged_tensor, (224, 224), mode='bilinear')
        lpips_score = loss_fn_alex(source_224, dragged_224).item()
    
    return lpips_score


def compute_mean_distance(source_image, dragged_image, points, prompt, dift, device):
    """Compute mean distance between target points and final handle positions"""
    source_image_PIL = Image.fromarray(source_image)
    dragged_image_PIL = Image.fromarray(dragged_image)
    dragged_image_PIL = dragged_image_PIL.resize(source_image_PIL.size, PIL.Image.BILINEAR)
    
    source_tensor = (PILToTensor()(source_image_PIL) / 255.0 - 0.5) * 2
    dragged_tensor = (PILToTensor()(dragged_image_PIL) / 255.0 - 0.5) * 2
    
    _, H, W = source_tensor.shape
    
    ft_source = dift.forward(source_tensor, prompt=prompt, t=261,
                             up_ft_index=1, ensemble_size=8)
    ft_source = F.interpolate(ft_source, (H, W), mode='bilinear')
    
    ft_dragged = dift.forward(dragged_tensor, prompt=prompt, t=261,
                             up_ft_index=1, ensemble_size=8)
    ft_dragged = F.interpolate(ft_dragged, (H, W), mode='bilinear')
    
    handle_points = []
    target_points = []
    for idx, point in enumerate(points):
        cur_point = torch.tensor([point[1], point[0]])  # row, col
        if idx % 2 == 0:
            handle_points.append(cur_point)
        else:
            target_points.append(cur_point)
    
    cos = torch.nn.CosineSimilarity(dim=1)
    all_dist = []
    for pt_idx in range(len(handle_points)):
        hp = handle_points[pt_idx]
        tp = target_points[pt_idx]
        
        num_channel = ft_source.size(1)
        src_vec = ft_source[0, :, int(hp[0]), int(hp[1])].view(1, num_channel, 1, 1)
        cos_map = cos(src_vec, ft_dragged).cpu().numpy()[0]
        max_rc = np.unravel_index(cos_map.argmax(), cos_map.shape)
        
        dist = (tp - torch.tensor(max_rc)).float().norm()
        all_dist.append(dist)
    
    return torch.tensor(all_dist).mean().item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare single vs multi-timestep")
    parser.add_argument('--lora_steps', type=int, default=80, help='number of lora fine-tuning steps')
    parser.add_argument('--inv_strength', type=float, default=0.7, help='inversion strength')
    parser.add_argument('--latent_lr', type=float, default=0.01, help='latent learning rate')
    parser.add_argument('--unet_feature_idx', type=int, default=3, help='feature idx of unet features')
    parser.add_argument('--lambda_reg', type=float, default=0.1, help='mask regularization weight')
    parser.add_argument('--n_pix_step', type=int, default=120, help='number of pixel update steps')
    parser.add_argument('--subset_size', type=int, default=None, help='limit number of samples (for testing)')
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Initialize evaluation models
    print("Initializing evaluation models...")
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)
    dift = SDFeaturizer('Manojb/stable-diffusion-2-1-base')
    seed_everything(42)  # For DIFT consistency
    
    all_category = [
        'art_work', 'land_scape', 'building_city_view', 'building_countryside_view',
        'animals', 'human_head', 'human_upper_body', 'human_full_body',
        'interior_design', 'other_objects',
    ]

    root_dir = 'drag_bench_data'
    lora_dir = 'drag_bench_lora'
    
    # Result directories
    baseline_dir = 'multi_timestep_baseline'
    multi_timestep_dir = 'multi_timestep_multi'
    
    for result_dir in [baseline_dir, multi_timestep_dir]:
        if not os.path.isdir(result_dir):
            os.mkdir(result_dir)
            for cat in all_category:
                os.mkdir(os.path.join(result_dir, cat))

    # Settings
    t_baseline = 35  # Single timestep
    t_set_multi = [30, 35, 40]  # Multi timestep
    
    # Metrics storage
    baseline_lpips = []
    baseline_md = []
    baseline_runtimes = []
    multi_lpips = []
    multi_md = []
    multi_runtimes = []
    
    sample_count = 0
    for cat in all_category:
        file_dir = os.path.join(root_dir, cat)
        for sample_name in os.listdir(file_dir):
            if sample_name == '.DS_Store':
                continue
            
            if args.subset_size and sample_count >= args.subset_size:
                break
            
            sample_path = os.path.join(file_dir, sample_name)
            print(f"\n{'='*60}")
            print(f"Processing {cat}/{sample_name}")
            print(f"{'='*60}")

            # Load data
            source_image = Image.open(os.path.join(sample_path, 'original_image.png'))
            source_image = np.array(source_image)
            
            with open(os.path.join(sample_path, 'meta_data.pkl'), 'rb') as f:
                meta_data = pickle.load(f)
            prompt = meta_data['prompt']
            mask = meta_data['mask']
            points = meta_data['points']

            lora_path = os.path.join(lora_dir, cat, sample_name, str(args.lora_steps))
            
            # Run baseline (single timestep t=35)
            print(f"\n[Baseline] Running single-timestep (t={t_baseline})...")
            try:
                baseline_image, baseline_runtime = run_drag_single_timestep(
                    source_image, mask, prompt, points, t_baseline,
                    args.inv_strength, args.lambda_reg, args.latent_lr,
                    args.unet_feature_idx, args.n_pix_step,
                    "runwayml/stable-diffusion-v1-5", "default", lora_path,
                    0, 10)
                baseline_runtimes.append(baseline_runtime)
                
                # Compute metrics
                lpips_score = compute_lpips(source_image, baseline_image, loss_fn_alex, device)
                baseline_lpips.append(lpips_score)
                if_score = 1.0 - lpips_score  # Image Fidelity
                
                md_score = compute_mean_distance(source_image, baseline_image, points, prompt, dift, device)
                baseline_md.append(md_score)
                
                print(f"  Runtime: {baseline_runtime:.2f}s")
                print(f"  LPIPS: {lpips_score:.4f}, IF (1-LPIPS): {if_score:.4f}")
                print(f"  Mean Distance: {md_score:.2f}px")
                
                # Save
                save_dir = os.path.join(baseline_dir, cat, sample_name)
                os.makedirs(save_dir, exist_ok=True)
                Image.fromarray(baseline_image).save(os.path.join(save_dir, 'dragged_image.png'))
            except Exception as e:
                print(f"  ERROR in baseline: {e}")
                continue

            # Run multi-timestep (t_set={30,35,40})
            print(f"\n[Multi-timestep] Running multi-timestep (t_set={t_set_multi})...")
            try:
                multi_image, multi_runtime = run_drag_multi_timestep(
                    source_image, mask, prompt, points, t_set_multi,
                    args.inv_strength, args.lambda_reg, args.latent_lr,
                    args.unet_feature_idx, args.n_pix_step,
                    "runwayml/stable-diffusion-v1-5", "default", lora_path,
                    0, 10)
                multi_runtimes.append(multi_runtime)
                
                # Compute metrics
                lpips_score = compute_lpips(source_image, multi_image, loss_fn_alex, device)
                multi_lpips.append(lpips_score)
                if_score = 1.0 - lpips_score
                
                md_score = compute_mean_distance(source_image, multi_image, points, prompt, dift, device)
                multi_md.append(md_score)
                
                print(f"  Runtime: {multi_runtime:.2f}s")
                print(f"  LPIPS: {lpips_score:.4f}, IF (1-LPIPS): {if_score:.4f}")
                print(f"  Mean Distance: {md_score:.2f}px")
                
                # Save
                save_dir = os.path.join(multi_timestep_dir, cat, sample_name)
                os.makedirs(save_dir, exist_ok=True)
                Image.fromarray(multi_image).save(os.path.join(save_dir, 'dragged_image.png'))
            except Exception as e:
                print(f"  ERROR in multi-timestep: {e}")
                continue
            
            sample_count += 1
        
        if args.subset_size and sample_count >= args.subset_size:
            break

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY RESULTS")
    print(f"{'='*60}")
    print(f"\nBaseline (Single-timestep t={t_baseline}):")
    print(f"  Average Mean Distance (MD): {np.mean(baseline_md):.2f}px")
    print(f"  Average Image Fidelity (IF = 1-LPIPS): {np.mean([1-x for x in baseline_lpips]):.4f}")
    print(f"  Average Runtime: {np.mean(baseline_runtimes):.2f}s")
    print(f"  Total samples: {len(baseline_md)}")
    
    print(f"\nMulti-timestep (t_set={t_set_multi}):")
    print(f"  Average Mean Distance (MD): {np.mean(multi_md):.2f}px")
    print(f"  Average Image Fidelity (IF = 1-LPIPS): {np.mean([1-x for x in multi_lpips]):.4f}")
    print(f"  Average Runtime: {np.mean(multi_runtimes):.2f}s")
    print(f"  Total samples: {len(multi_md)}")
    
    print(f"\nComparison:")
    print(f"  MD improvement: {np.mean(baseline_md) - np.mean(multi_md):.2f}px ({((np.mean(baseline_md) - np.mean(multi_md)) / np.mean(baseline_md) * 100):.1f}%)")
    print(f"  IF improvement: {np.mean([1-x for x in multi_lpips]) - np.mean([1-x for x in baseline_lpips]):.4f}")
    print(f"  Runtime overhead: {np.mean(multi_runtimes) - np.mean(baseline_runtimes):.2f}s ({((np.mean(multi_runtimes) - np.mean(baseline_runtimes)) / np.mean(baseline_runtimes) * 100):.1f}%)")
    print(f"{'='*60}")

