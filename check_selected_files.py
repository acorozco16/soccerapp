#!/usr/bin/env python3
"""
Check what files were actually selected in the recent dataset
"""

import zipfile
from pathlib import Path
from collections import Counter

def analyze_selected_files():
    """Analyze what files were selected as 'recent'"""
    
    zip_file = Path("/Users/andreworozco/soccer app/training_data/keepups_dataset_recent_3700.zip")
    if not zip_file.exists():
        print("❌ Recent dataset zip not found")
        return
    
    print("🔍 Analyzing selected 'recent' files...")
    
    # Extract filenames from zip
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        image_files = [f for f in zip_ref.namelist() if f.endswith('.jpg')]
    
    print(f"📊 Total images in recent dataset: {len(image_files)}")
    
    # Analyze filename patterns
    patterns = Counter()
    prefixes = Counter()
    
    for img_path in image_files:
        filename = Path(img_path).name
        
        # Count different patterns
        if 'frame_' in filename:
            patterns['frame_'] += 1
        if '_frame_' in filename:
            patterns['_frame_'] += 1
        if '2024' in filename:
            patterns['2024'] += 1
        if 'yt_' in filename:
            patterns['youtube'] += 1
        
        # Get filename prefixes (first part before _)
        prefix = filename.split('_')[0][:10]  # First 10 chars of prefix
        prefixes[prefix] += 1
    
    print("\n📊 Pattern Analysis:")
    for pattern, count in patterns.most_common():
        print(f"   {pattern}: {count} files")
    
    print(f"\n📊 Top filename prefixes:")
    for prefix, count in prefixes.most_common(10):
        print(f"   {prefix}*: {count} files")
    
    # Show some sample filenames
    print(f"\n📋 Sample filenames (first 10):")
    for i, img_path in enumerate(image_files[:10]):
        filename = Path(img_path).name
        print(f"   {i+1}. {filename}")
    
    print(f"\n💭 Does this look like your recent 3,700 batch?")
    print(f"   - Do you recognize these filename patterns?")
    print(f"   - Are these the files you labeled in the last 24 hours?")

if __name__ == "__main__":
    analyze_selected_files()