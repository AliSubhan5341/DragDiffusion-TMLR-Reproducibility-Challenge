#!/usr/bin/env python3
"""
Script to copy and rename sample images from drag_bench_evaluation result directories.
Creates a new folder structure with renamed images from each category.
Also copies original and drag images from drag_bench_data.
"""

import os
import shutil
from pathlib import Path

# Base paths
EVALUATION_DIR = Path("/shared/home/ali.subhan/DragDiffusion/drag_bench_evaluation")
OUTPUT_DIR = EVALUATION_DIR / "sampled_results"
DRAG_BENCH_DATA_DIR = EVALUATION_DIR / "drag_bench_data"

# Categories to process
CATEGORIES = [
    "animals",
    "art_work",
    "building_city_view",
    "building_countryside_view",
    "human_full_body",
    "human_head",
    "human_upper_body",
    "interior_design",
    "land_scape",
    "other_objects"
]

# Directories to skip (not result directories)
SKIP_DIRS = {
    "drag_bench_data",
    "drag_bench_lora",
    "__pycache__",
    "sampled_results"
}

# Number of samples to copy from each category
NUM_SAMPLES = 3

# Store which samples are used per category (to copy originals later)
used_samples = {}  # {category: [sample_folder_names]}


def get_result_directories():
    """Get all result directories from evaluation folder."""
    result_dirs = []
    for item in EVALUATION_DIR.iterdir():
        if item.is_dir() and item.name not in SKIP_DIRS:
            result_dirs.append(item)
    return sorted(result_dirs)


def get_sample_folders_for_category(category):
    """
    Get the first NUM_SAMPLES sample folder names for a category.
    Uses the first result directory to determine which samples to use.
    """
    result_dirs = get_result_directories()
    
    for result_dir in result_dirs:
        category_path = result_dir / category
        if category_path.exists():
            sample_folders = sorted([f.name for f in category_path.iterdir() if f.is_dir()])
            if sample_folders:
                return sample_folders[:NUM_SAMPLES]
    
    return []


def copy_images_from_category(result_dir, category, output_category_dir):
    """
    Copy NUM_SAMPLES images from a specific category in a result directory.
    
    Args:
        result_dir: Path to the result directory
        category: Category name (e.g., 'animals')
        output_category_dir: Path to output category directory
    """
    category_path = result_dir / category
    
    if not category_path.exists():
        print(f"  ⚠️  Category '{category}' not found in {result_dir.name}")
        return
    
    # Get all sample folders in this category
    sample_folders = sorted([f for f in category_path.iterdir() if f.is_dir()])
    
    if not sample_folders:
        print(f"  ⚠️  No samples found in {result_dir.name}/{category}")
        return
    
    # Process first NUM_SAMPLES samples
    samples_to_process = sample_folders[:NUM_SAMPLES]
    
    # Store sample names for later (to copy originals)
    if category not in used_samples:
        used_samples[category] = [s.name for s in samples_to_process]
    
    for idx, sample_folder in enumerate(samples_to_process, start=1):
        # Look for dragged_image.png in the sample folder
        image_path = sample_folder / "dragged_image.png"
        
        if not image_path.exists():
            print(f"  ⚠️  Image not found: {image_path}")
            continue
        
        # Create output filename: {number}_{category}_{result_dir_name}.png
        output_filename = f"{idx}_{category}_{result_dir.name}.png"
        output_path = output_category_dir / output_filename
        
        # Copy the image
        shutil.copy2(image_path, output_path)
        print(f"  ✓ Copied: {output_filename}")


def copy_original_and_drag_images():
    """
    Copy original_image.png and user_drag.png from drag_bench_data
    for all samples that were used.
    """
    print("\n" + "=" * 70)
    print("📂 Copying original and drag images from drag_bench_data")
    print("=" * 70 + "\n")
    
    for category in CATEGORIES:
        if category not in used_samples:
            print(f"  ⚠️  No samples recorded for {category}")
            continue
        
        output_category_dir = OUTPUT_DIR / category
        sample_names = used_samples[category]
        
        print(f"📂 Category: {category}")
        
        for idx, sample_name in enumerate(sample_names, start=1):
            sample_path = DRAG_BENCH_DATA_DIR / category / sample_name
            
            if not sample_path.exists():
                print(f"  ⚠️  Sample not found in drag_bench_data: {sample_path}")
                continue
            
            # Copy original_image.png
            original_path = sample_path / "original_image.png"
            if original_path.exists():
                output_filename = f"{idx}_{category}_original.png"
                output_path = output_category_dir / output_filename
                shutil.copy2(original_path, output_path)
                print(f"  ✓ Copied: {output_filename}")
            else:
                print(f"  ⚠️  Original image not found: {original_path}")
            
            # Copy user_drag.png
            drag_path = sample_path / "user_drag.png"
            if drag_path.exists():
                output_filename = f"{idx}_{category}_user_drag.png"
                output_path = output_category_dir / output_filename
                shutil.copy2(drag_path, output_path)
                print(f"  ✓ Copied: {output_filename}")
            else:
                print(f"  ⚠️  User drag image not found: {drag_path}")
        
        print()  # Empty line between categories


def main():
    """Main function to organize and copy images."""
    print("=" * 70)
    print("Drag Diffusion Evaluation - Sample Image Copy Script")
    print("=" * 70)
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"\n📁 Output directory: {OUTPUT_DIR}\n")
    
    # Create category directories
    for category in CATEGORIES:
        category_dir = OUTPUT_DIR / category
        category_dir.mkdir(exist_ok=True)
    
    # Get all result directories
    result_dirs = get_result_directories()
    print(f"Found {len(result_dirs)} result directories to process\n")
    
    # Process each result directory
    for result_dir in result_dirs:
        print(f"📂 Processing: {result_dir.name}")
        
        for category in CATEGORIES:
            output_category_dir = OUTPUT_DIR / category
            copy_images_from_category(result_dir, category, output_category_dir)
        
        print()  # Empty line between result directories
    
    # Copy original and drag images from drag_bench_data
    copy_original_and_drag_images()
    
    print("=" * 70)
    print("✅ Image copying completed!")
    print(f"📊 Results saved to: {OUTPUT_DIR}")
    print("=" * 70)
    
    # Print summary statistics
    print("\n📈 Summary:")
    for category in CATEGORIES:
        category_dir = OUTPUT_DIR / category
        num_images = len(list(category_dir.glob("*.png")))
        print(f"  {category:30s}: {num_images:3d} images")
    
    # Print sample info
    print("\n📋 Samples used per category:")
    for category, samples in used_samples.items():
        print(f"  {category}: {samples}")


if __name__ == "__main__":
    main()
