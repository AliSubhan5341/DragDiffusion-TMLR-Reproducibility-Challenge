#!/usr/bin/env python
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
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *************************************************************************

"""
Compute Image Fidelity (IF) for edited images using LPIPS.

For a set of original–edited image pairs, this script:
  - computes LPIPS(original, edited) for each pair
  - converts it to Image Fidelity as IF = 1 - LPIPS
  - reports the mean and standard deviation of IF across all evaluated pairs

Example:
    python run_eval_image_fidelity.py \\
        --original_dir path/to/originals \\
        --edited_dir   path/to/edited \\
        --num_samples  10

Both directories must contain images with matching filenames. The first
`num_samples` matching filenames (sorted lexicographically) will be used.
"""

import argparse
import os
from typing import List

import numpy as np
from PIL import Image

import torch
from einops import rearrange
import lpips


def preprocess_image(image: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Preprocess a HWC uint8 image array into a BCHW float tensor in [-1, 1],
    matching the behavior used elsewhere in DragDiffusion (e.g., LPIPS eval).
    """
    image = torch.from_numpy(image).float() / 127.5 - 1.0  # [-1, 1]
    image = rearrange(image, "h w c -> 1 c h w")
    image = image.to(device)
    return image


def list_matching_filenames(original_dir: str, edited_dir: str) -> List[str]:
    """Return sorted list of filenames that exist in both dirs."""
    orig_files = {f for f in os.listdir(original_dir)
                  if os.path.isfile(os.path.join(original_dir, f))}
    edited_files = {f for f in os.listdir(edited_dir)
                    if os.path.isfile(os.path.join(edited_dir, f))}
    common = sorted(orig_files.intersection(edited_files))
    return common


def main():
    parser = argparse.ArgumentParser(
        description="Compute Image Fidelity (IF = 1 - LPIPS) for image pairs."
    )
    parser.add_argument(
        "--original_dir",
        type=str,
        required=True,
        help="Directory containing original images.",
    )
    parser.add_argument(
        "--edited_dir",
        type=str,
        required=True,
        help="Directory containing edited images.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of image pairs to evaluate (default: 10).",
    )
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # LPIPS metric (Alex backbone, same as in other evaluation code)
    loss_fn_alex = lpips.LPIPS(net="alex").to(device)

    # Find matching filenames between original and edited directories
    matching_filenames = list_matching_filenames(args.original_dir, args.edited_dir)
    if not matching_filenames:
        raise ValueError(
            f"No matching filenames found between "
            f"'{args.original_dir}' and '{args.edited_dir}'."
        )

    num_pairs = min(args.num_samples, len(matching_filenames))
    matching_filenames = matching_filenames[:num_pairs]

    print(f"Evaluating Image Fidelity on {num_pairs} image pairs.")
    print(f"Original dir: {args.original_dir}")
    print(f"Edited dir:   {args.edited_dir}")

    all_if_values: List[float] = []

    for fname in matching_filenames:
        orig_path = os.path.join(args.original_dir, fname)
        edited_path = os.path.join(args.edited_dir, fname)

        # Load images
        orig_img = Image.open(orig_path).convert("RGB")
        edited_img = Image.open(edited_path).convert("RGB")

        # Resize edited image to match original resolution if needed
        if edited_img.size != orig_img.size:
            edited_img = edited_img.resize(orig_img.size, Image.BILINEAR)

        # Preprocess to tensors
        orig_tensor = preprocess_image(np.array(orig_img), device)
        edited_tensor = preprocess_image(np.array(edited_img), device)

        # Compute LPIPS distance
        with torch.no_grad():
            lpips_val = loss_fn_alex(orig_tensor, edited_tensor).item()

        # Image Fidelity as IF = 1 - LPIPS
        if_val = 1.0 - lpips_val
        all_if_values.append(if_val)

        print(f"{fname}: LPIPS = {lpips_val:.6f}, IF = {if_val:.6f}")

    all_if_values_np = np.array(all_if_values, dtype=np.float32)
    mean_if = float(all_if_values_np.mean())
    std_if = float(all_if_values_np.std(ddof=0))

    print("--------------------------------------------------")
    print(f"Image Fidelity (IF = 1 - LPIPS) over {num_pairs} samples:")
    print(f"Mean IF: {mean_if:.6f}")
    print(f"Std IF:  {std_if:.6f}")


if __name__ == "__main__":
    main()





