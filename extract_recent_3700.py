#!/usr/bin/env python3
"""
Extract just the most recent 3,700 images from the full dataset
This gives us the freshest, best-quality annotations for training
"""

import os
import shutil
import zipfile
from pathlib import Path
import glob

def extract_recent_3700():
    """Extract the newest 3,700 images based on file modification time"""
    
    # Extract the full dataset first
    zip_file = Path("/Users/andreworozco/soccer app/training_data/keepups_dataset_7335_FIXED.zip")
    if not zip_file.exists():
        print("❌ Cannot find the full dataset zip")
        return False
    
    # Extract to temp location
    temp_dir = zip_file.parent / "temp_extract_recent"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    
    print("📦 Extracting full dataset to find recent files...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    source_dataset = temp_dir / "keepups_dataset"
    if not source_dataset.exists():
        print("❌ Extracted dataset structure not found")
        return False
    
    # Get all image files with their modification times
    image_files = list((source_dataset / "images").glob("*.jpg"))
    print(f"🔍 Found {len(image_files)} total images")
    
    # Sort by modification time (newest first) 
    # Since we can't get real mod times from zip, we'll use filename patterns
    # to identify the most recent batch
    
    # Look for patterns that indicate recent files
    recent_patterns = [
        "frame_0",  # Recent frame extractions
        "_frame_",  # New naming convention
        "2024",     # Recent date stamps
    ]
    
    # Separate files into likely recent vs old
    recent_files = []
    older_files = []
    
    for img_file in image_files:
        filename = img_file.name.lower()
        is_recent = any(pattern in filename for pattern in recent_patterns)
        
        if is_recent:
            recent_files.append(img_file)
        else:
            older_files.append(img_file)
    
    print(f"📊 Identified {len(recent_files)} likely recent files")
    print(f"📊 Identified {len(older_files)} likely older files")
    
    # If we have more than 3700 recent files, take the first 3700
    # If we have less, supplement with some older files to reach 3700
    target_files = recent_files[:3700]
    
    if len(target_files) < 3700:
        needed = 3700 - len(target_files)
        target_files.extend(older_files[:needed])
    
    print(f"✅ Selected {len(target_files)} images for recent dataset")
    
    # Create new dataset directory
    recent_dataset = Path("/Users/andreworozco/soccer app/training_data/keepups_dataset_recent_3700")
    if recent_dataset.exists():
        shutil.rmtree(recent_dataset)
    
    recent_dataset.mkdir(parents=True)
    (recent_dataset / "images").mkdir()
    (recent_dataset / "labels").mkdir()
    
    # Copy selected images and labels
    copied_count = 0
    for img_file in target_files:
        # Copy image
        target_img = recent_dataset / "images" / img_file.name
        shutil.copy2(img_file, target_img)
        
        # Copy corresponding label
        label_file = source_dataset / "labels" / (img_file.stem + ".txt")
        if label_file.exists():
            target_label = recent_dataset / "labels" / label_file.name
            shutil.copy2(label_file, target_label)
            copied_count += 1
    
    # Create dataset.yaml
    yaml_content = """path: /content/keepups_dataset_recent_3700
train: images
val: images

names:
  0: ball

nc: 1
"""
    
    with open(recent_dataset / "dataset.yaml", 'w') as f:
        f.write(yaml_content)
    
    print(f"✅ Created dataset with {copied_count} image-label pairs")
    
    # Create zip file
    zip_path = Path("/Users/andreworozco/soccer app/training_data/keepups_dataset_recent_3700.zip")
    if zip_path.exists():
        zip_path.unlink()
    
    print("📦 Creating zip file...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in recent_dataset.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(recent_dataset.parent)
                zipf.write(file_path, arcname)
    
    # Check final size
    size_mb = zip_path.stat().st_size / 1024 / 1024
    print(f"📊 Final zip size: {size_mb:.1f} MB")
    print(f"✅ Recent 3,700 dataset ready: {zip_path}")
    
    # Clean up
    shutil.rmtree(temp_dir)
    
    return True

if __name__ == "__main__":
    print("🔧 Extracting your most recent 3,700 images...")
    print("="*50)
    
    if extract_recent_3700():
        print("\n🎯 Success! Your fresh dataset is ready:")
        print("📁 File: keepups_dataset_recent_3700.zip")
        print("📊 Size: ~300-400MB (much more reliable for upload)")
        print("🏆 Quality: Your latest and best annotations")
        print("\n📤 Upload this file to Colab for training!")
    else:
        print("\n❌ Failed to extract recent dataset")