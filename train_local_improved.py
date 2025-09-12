#!/usr/bin/env python3
"""
Train the improved model locally on Mac (slower but free)
This uses the fixed dataset and runs the same training as Colab
"""

import os
import sys
from pathlib import Path

def check_requirements():
    """Check if we have ultralytics installed"""
    try:
        from ultralytics import YOLO
        print("✅ ultralytics is available")
        return True
    except ImportError:
        print("❌ ultralytics not installed")
        print("Run: pip3 install ultralytics")
        return False

def prepare_dataset():
    """Prepare the fixed dataset for training"""
    dataset_dir = Path(__file__).parent / "training_data" / "keepups_dataset_fixed"
    zip_file = Path(__file__).parent / "training_data" / "keepups_dataset_7335_FIXED.zip"
    
    if not zip_file.exists():
        print(f"❌ Fixed dataset not found: {zip_file}")
        print("Make sure you have the keepups_dataset_7335_FIXED.zip file")
        return None
    
    # Extract if needed
    if not dataset_dir.exists():
        print("📦 Extracting fixed dataset...")
        import zipfile
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir.parent)
        
        # Rename extracted folder
        extracted_dir = dataset_dir.parent / "keepups_dataset"
        if extracted_dir.exists():
            extracted_dir.rename(dataset_dir)
    
    # Update dataset.yaml for local paths
    dataset_yaml = dataset_dir / "dataset.yaml"
    yaml_content = f"""path: {dataset_dir.absolute()}
train: images
val: images

names:
  0: ball

nc: 1
"""
    
    with open(dataset_yaml, 'w') as f:
        f.write(yaml_content)
    
    print(f"✅ Dataset ready at: {dataset_dir}")
    return dataset_yaml

def train_model_locally(dataset_yaml):
    """Train the model locally (slower but free)"""
    from ultralytics import YOLO
    
    print("🚀 Starting LOCAL training...")
    print("⏱️ Expected time: 8-12 hours (CPU) or 4-6 hours (if you have Metal GPU)")
    print("💡 You can stop and resume training anytime with Ctrl+C")
    
    # Load pretrained model
    model = YOLO('yolov8n.pt')
    
    # Train with local-optimized settings
    try:
        results = model.train(
            data=str(dataset_yaml),
            epochs=100,              # Full training
            imgsz=640,              # Standard size
            batch=4,                # Smaller batch for Mac memory
            patience=25,            # More patience for slower training
            save=True,
            cache=False,            # Don't cache to save memory
            device='mps' if sys.platform == 'darwin' else 'cpu',  # Use Mac GPU if available
            workers=2,              # Conservative for Mac
            project='runs/train',
            name='local_improved_soccer_ball',
            exist_ok=True
        )
        
        print("🎉 Training completed successfully!")
        
        # Copy the best model
        best_model = Path("runs/train/local_improved_soccer_ball/weights/best.pt")
        if best_model.exists():
            target = Path(__file__).parent / "models" / "soccer_ball_trained_v2.pt"
            import shutil
            shutil.copy2(best_model, target)
            print(f"✅ New model saved as: {target}")
            print("📊 Ready to test the improved model!")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⏸️ Training interrupted by user")
        print("💡 You can resume later by running this script again")
        return False
    except Exception as e:
        print(f"❌ Training error: {e}")
        return False

if __name__ == "__main__":
    print("🖥️ Local Soccer Ball Model Training")
    print("="*50)
    
    if not check_requirements():
        sys.exit(1)
    
    dataset_yaml = prepare_dataset()
    if dataset_yaml is None:
        sys.exit(1)
    
    success = train_model_locally(dataset_yaml)
    
    if success:
        print("\n🎯 Training Complete! Next steps:")
        print("1. Test the new model with: python3 test_current_model.py")
        print("2. Compare performance with your original model")
        print("3. Deploy the improved model to your app")
    else:
        print("\n⚠️ Training incomplete. You can:")
        print("1. Resume training by running this script again")
        print("2. Try Kaggle or wait for Colab quota reset")
        print("3. Consider Colab Pro for faster training")