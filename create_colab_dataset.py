#!/usr/bin/env python3
"""
Create a smaller, more reliable dataset for Colab upload
This takes a subset of your fixed dataset to avoid upload issues
"""

import os
import shutil
import zipfile
from pathlib import Path
import random

def create_colab_dataset():
    """Create a smaller dataset (2000 images) for reliable Colab upload"""
    
    # Source: your fixed dataset
    source_dataset = Path("/Users/andreworozco/soccer app/training_data/temp_check/keepups_dataset")
    if not source_dataset.exists():
        # Try to extract from the fixed zip
        zip_file = Path("/Users/andreworozco/soccer app/training_data/keepups_dataset_7335_FIXED.zip")
        if zip_file.exists():
            print("📦 Extracting source dataset...")
            temp_dir = zip_file.parent / "temp_extract"
            temp_dir.mkdir(exist_ok=True)
            
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            source_dataset = temp_dir / "keepups_dataset"
        else:
            print("❌ Cannot find source dataset")
            return False
    
    # Create smaller dataset directory
    small_dataset = Path("/Users/andreworozco/soccer app/training_data/keepups_dataset_colab")
    if small_dataset.exists():
        shutil.rmtree(small_dataset)
    
    small_dataset.mkdir(parents=True)
    (small_dataset / "images").mkdir()
    (small_dataset / "labels").mkdir()
    
    # Get all image files
    image_files = list((source_dataset / "images").glob("*.jpg"))
    print(f"🔍 Found {len(image_files)} total images")
    
    # Randomly select 2000 images for faster upload
    selected_images = random.sample(image_files, min(2000, len(image_files)))
    print(f"📊 Selected {len(selected_images)} images for Colab")
    
    # Copy selected images and corresponding labels
    copied_count = 0
    for img_file in selected_images:
        # Copy image
        target_img = small_dataset / "images" / img_file.name
        shutil.copy2(img_file, target_img)
        
        # Copy corresponding label (if exists)
        label_file = source_dataset / "labels" / (img_file.stem + ".txt")
        if label_file.exists():
            target_label = small_dataset / "labels" / label_file.name
            shutil.copy2(label_file, target_label)
            copied_count += 1
    
    # Create dataset.yaml
    yaml_content = """path: /content/keepups_dataset_colab
train: images
val: images

names:
  0: ball

nc: 1
"""
    
    with open(small_dataset / "dataset.yaml", 'w') as f:
        f.write(yaml_content)
    
    print(f"✅ Created dataset with {copied_count} image-label pairs")
    
    # Create zip file
    zip_path = Path("/Users/andreworozco/soccer app/training_data/keepups_dataset_colab.zip")
    if zip_path.exists():
        zip_path.unlink()
    
    print("📦 Creating zip file...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in small_dataset.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(small_dataset.parent)
                zipf.write(file_path, arcname)
    
    # Check final size
    size_mb = zip_path.stat().st_size / 1024 / 1024
    print(f"📊 Final zip size: {size_mb:.1f} MB")
    print(f"✅ Dataset ready for upload: {zip_path}")
    
    # Clean up temp directory
    temp_dir = zip_path.parent / "temp_extract"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    return True

if __name__ == "__main__":
    print("🔧 Creating reliable Colab dataset...")
    print("="*50)
    
    if create_colab_dataset():
        print("\n🎯 Success! Next steps:")
        print("1. Upload 'keepups_dataset_colab.zip' to Colab (much smaller)")
        print("2. This will train faster and upload more reliably")
        print("3. You'll still get excellent results with 2000 images")
    else:
        print("\n❌ Failed to create dataset")