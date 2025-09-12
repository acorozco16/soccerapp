#!/usr/bin/env python3
"""
Fix corrupted YOLO label files by converting literal \n to actual newlines
This fixes the error: could not convert string to float: '0.050000\\n'
"""

import os
import glob
from pathlib import Path

def fix_label_files(dataset_path):
    """Fix all label files in the dataset"""
    labels_dir = Path(dataset_path) / "labels"
    
    if not labels_dir.exists():
        print(f"❌ Labels directory not found: {labels_dir}")
        return False
        
    label_files = list(labels_dir.glob("*.txt"))
    print(f"🔍 Found {len(label_files)} label files to fix")
    
    fixed_count = 0
    error_count = 0
    
    for label_file in label_files:
        try:
            # Read the corrupted file
            with open(label_file, 'r') as f:
                content = f.read().strip()
            
            # Check if it needs fixing (has literal \n)
            if '\\n' in content:
                # Fix: Replace literal \n with actual newlines
                fixed_content = content.replace('\\n', '\n')
                
                # Remove any trailing newlines and normalize
                lines = [line.strip() for line in fixed_content.split('\n') if line.strip()]
                final_content = '\n'.join(lines)
                
                # Write back the fixed content
                with open(label_file, 'w') as f:
                    f.write(final_content)
                    if lines:  # Add final newline only if file has content
                        f.write('\n')
                
                fixed_count += 1
                if fixed_count <= 5:  # Show first 5 examples
                    print(f"✅ Fixed: {label_file.name}")
                    print(f"   Before: {repr(content[:50])}...")
                    print(f"   After:  {repr(final_content[:50])}...")
                    print()
            
        except Exception as e:
            print(f"❌ Error fixing {label_file.name}: {e}")
            error_count += 1
    
    print(f"\n📊 Results:")
    print(f"   ✅ Fixed: {fixed_count} files")
    print(f"   ⚠️  Skipped (already clean): {len(label_files) - fixed_count - error_count} files")
    print(f"   ❌ Errors: {error_count} files")
    
    return fixed_count > 0

def validate_fixed_files(dataset_path):
    """Validate that the fixed files are now in correct YOLO format"""
    labels_dir = Path(dataset_path) / "labels"
    sample_files = list(labels_dir.glob("*.txt"))[:5]
    
    print(f"\n🧪 Validating {len(sample_files)} sample files:")
    
    for label_file in sample_files:
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            print(f"\n📄 {label_file.name}:")
            for i, line in enumerate(lines[:3]):  # Show first 3 lines
                parts = line.strip().split()
                if len(parts) == 5:
                    try:
                        class_id = int(parts[0])
                        coords = [float(x) for x in parts[1:5]]
                        print(f"   Line {i+1}: ✅ Valid - Class {class_id}, coords {coords}")
                    except ValueError as e:
                        print(f"   Line {i+1}: ❌ Invalid - {e}")
                else:
                    print(f"   Line {i+1}: ❌ Wrong format - {len(parts)} parts instead of 5")
            
            if len(lines) > 3:
                print(f"   ... and {len(lines) - 3} more lines")
                
        except Exception as e:
            print(f"❌ Error validating {label_file.name}: {e}")

if __name__ == "__main__":
    # Fix the extracted dataset
    dataset_path = "/Users/andreworozco/soccer app/training_data/temp_check/keepups_dataset"
    
    print("🔧 Fixing YOLO label file format issues...")
    print("="*50)
    
    if fix_label_files(dataset_path):
        print("\n🎯 Label files have been fixed!")
        validate_fixed_files(dataset_path)
        
        print("\n📦 Next steps:")
        print("1. Recreate the zip file: cd temp_check && zip -r ../keepups_dataset_7335_FIXED.zip keepups_dataset/")
        print("2. Upload the FIXED zip to Colab")
        print("3. Retry training - it should work now!")
        
    else:
        print("\n⚠️  No files needed fixing or errors occurred")