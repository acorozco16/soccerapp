#!/usr/bin/env python3
"""
Split the large dataset into smaller chunks for reliable Colab upload
"""

import zipfile
import os
from pathlib import Path

def split_dataset():
    """Split the 615MB dataset into 2-3 smaller chunks"""
    
    source_zip = Path("/Users/andreworozco/soccer app/training_data/keepups_dataset_7335_FIXED.zip")
    if not source_zip.exists():
        print("❌ Source dataset not found")
        return False
    
    # Extract the dataset first
    extract_dir = source_zip.parent / "temp_split"
    if extract_dir.exists():
        import shutil
        shutil.rmtree(extract_dir)
    extract_dir.mkdir()
    
    print("📦 Extracting full dataset...")
    with zipfile.ZipFile(source_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    dataset_dir = extract_dir / "keepups_dataset"
    if not dataset_dir.exists():
        print("❌ Dataset structure not found after extraction")
        return False
    
    # Get all files
    images = list((dataset_dir / "images").glob("*.jpg"))
    labels = list((dataset_dir / "labels").glob("*.txt"))
    
    print(f"📊 Found {len(images)} images and {len(labels)} labels")
    
    # Split into 3 chunks of ~2500 images each
    chunk_size = len(images) // 3
    chunks = [
        images[0:chunk_size],
        images[chunk_size:2*chunk_size], 
        images[2*chunk_size:]
    ]
    
    output_dir = source_zip.parent
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n📦 Creating chunk {i} with {len(chunk)} images...")
        
        # Create chunk directory
        chunk_dir = extract_dir / f"keepups_dataset_chunk_{i}"
        chunk_dir.mkdir()
        (chunk_dir / "images").mkdir()
        (chunk_dir / "labels").mkdir()
        
        # Copy images and labels for this chunk
        for img_file in chunk:
            # Copy image
            import shutil
            shutil.copy2(img_file, chunk_dir / "images" / img_file.name)
            
            # Copy corresponding label
            label_file = dataset_dir / "labels" / (img_file.stem + ".txt")
            if label_file.exists():
                shutil.copy2(label_file, chunk_dir / "labels" / label_file.name)
        
        # Create dataset.yaml for this chunk
        yaml_content = f"""path: /content/keepups_dataset_chunk_{i}
train: images
val: images

names:
  0: ball

nc: 1
"""
        with open(chunk_dir / "dataset.yaml", 'w') as f:
            f.write(yaml_content)
        
        # Create zip for this chunk
        zip_path = output_dir / f"keepups_dataset_chunk_{i}.zip"
        if zip_path.exists():
            zip_path.unlink()
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in chunk_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(extract_dir)
                    zipf.write(file_path, arcname)
        
        size_mb = zip_path.stat().st_size / 1024 / 1024
        print(f"✅ Chunk {i}: {size_mb:.1f} MB - {zip_path.name}")
    
    # Clean up
    import shutil
    shutil.rmtree(extract_dir)
    
    print(f"\n🎯 Success! Upload these 3 files to Colab:")
    print(f"   1. keepups_dataset_chunk_1.zip")
    print(f"   2. keepups_dataset_chunk_2.zip") 
    print(f"   3. keepups_dataset_chunk_3.zip")
    print(f"\n📝 Each file is ~200MB - much more reliable!")
    
    return True

if __name__ == "__main__":
    print("🔧 Splitting dataset into reliable chunks...")
    print("="*50)
    split_dataset()