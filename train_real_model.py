#!/usr/bin/env python3
"""
Train a new model using real video data combined with YouTube data
"""

from ultralytics import YOLO
import sys
from pathlib import Path

def train_real_model():
    """Train model on mixed dataset with real video data"""
    
    print("🎯 Training model with real video data + YouTube data")
    
    # Check if datasets exist
    mixed_dataset = Path("training_data/mixed_dataset/dataset.yaml")
    real_dataset = Path("training_data/real_dataset/dataset.yaml")
    
    if not mixed_dataset.exists():
        print(f"❌ Mixed dataset not found: {mixed_dataset}")
        return False
    
    try:
        # Load pretrained model
        model = YOLO('yolov8n.pt')
        print("✅ Loaded YOLOv8 nano pretrained model")
        
        # Train on mixed dataset (real + YouTube)
        print(f"🚀 Starting training on mixed dataset...")
        print(f"   Dataset: {mixed_dataset}")
        print(f"   Real data: 55 images (25 with ball)")
        print(f"   YouTube data: 500 images")
        
        results = model.train(
            data=str(mixed_dataset),
            epochs=20,  # Fewer epochs for faster training
            imgsz=640,
            batch=8,    # Smaller batch for mixed data
            name='real_soccer_detector',
            project='training_data/experiments',
            save_period=5,
            patience=8,
            lr0=0.001,  # Lower learning rate for fine-tuning
            warmup_epochs=3
        )
        
        print("✅ Mixed model training completed!")
        
        # Also train a model on just real data
        print(f"\n🎯 Training model on ONLY real data...")
        
        model_real = YOLO('yolov8n.pt')
        
        results_real = model_real.train(
            data=str(real_dataset),
            epochs=50,  # More epochs for small dataset
            imgsz=640,
            batch=4,    # Small batch for small dataset
            name='real_only_detector',
            project='training_data/experiments',
            save_period=10,
            patience=15,
            lr0=0.01,   # Higher learning rate for small dataset
            warmup_epochs=5
        )
        
        print("✅ Real-only model training completed!")
        
        # Print model paths
        mixed_model_path = Path("training_data/experiments/real_soccer_detector/weights/best.pt")
        real_model_path = Path("training_data/experiments/real_only_detector/weights/best.pt")
        
        print(f"\n📍 Model locations:")
        print(f"   Mixed model: {mixed_model_path}")
        print(f"   Real-only model: {real_model_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def test_models():
    """Test both new models on the original video"""
    
    print("\n🧪 Testing new models on original video")
    
    from ultralytics import YOLO
    import cv2
    
    # Model paths
    mixed_model = "training_data/experiments/real_soccer_detector/weights/best.pt"
    real_model = "training_data/experiments/real_only_detector/weights/best.pt"
    original_model = "training_data/experiments/soccer_ball_detector/weights/best.pt"
    
    video_path = "uploads/raw/20250726_075501_a984b9ba.mp4"
    
    models_to_test = [
        ("Original YouTube Model", original_model),
        ("Mixed Data Model", mixed_model),
        ("Real Data Only Model", real_model)
    ]
    
    results_summary = {}
    
    for model_name, model_path in models_to_test:
        if not Path(model_path).exists():
            print(f"⏭️ Skipping {model_name} - model not found")
            continue
        
        print(f"\n🔍 Testing {model_name}")
        print(f"   Model: {model_path}")
        
        try:
            model = YOLO(model_path)
            cap = cv2.VideoCapture(video_path)
            
            frame_count = 0
            detections = 0
            high_conf_detections = 0
            confidence_scores = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Test every 5th frame
                if frame_count % 5 != 0:
                    continue
                
                # Run detection
                results = model(frame, verbose=False)
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None and len(boxes) > 0:
                        detections += 1
                        
                        for box in boxes:
                            conf = float(box.conf[0])
                            confidence_scores.append(conf)
                            
                            if conf > 0.5:
                                high_conf_detections += 1
            
            cap.release()
            
            # Calculate metrics
            processed_frames = frame_count // 5
            detection_rate = (detections / processed_frames) * 100 if processed_frames > 0 else 0
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            max_confidence = max(confidence_scores) if confidence_scores else 0
            
            results_summary[model_name] = {
                'detection_rate': detection_rate,
                'avg_confidence': avg_confidence,
                'max_confidence': max_confidence,
                'high_conf_detections': high_conf_detections,
                'total_detections': detections
            }
            
            print(f"   Detection rate: {detection_rate:.1f}%")
            print(f"   Avg confidence: {avg_confidence:.3f}")
            print(f"   Max confidence: {max_confidence:.3f}")
            print(f"   High confidence (>0.5): {high_conf_detections}")
            
        except Exception as e:
            print(f"   ❌ Error testing {model_name}: {e}")
    
    # Summary comparison
    print(f"\n📊 MODEL COMPARISON SUMMARY:")
    print("=" * 60)
    for model_name, metrics in results_summary.items():
        print(f"{model_name}:")
        print(f"  Detection Rate: {metrics['detection_rate']:.1f}%")
        print(f"  Avg Confidence: {metrics['avg_confidence']:.3f}")
        print(f"  Max Confidence: {metrics['max_confidence']:.3f}")
        print(f"  High Confidence Detections: {metrics['high_conf_detections']}")
        print()
    
    # Determine best model
    if results_summary:
        best_model = max(results_summary.items(), 
                        key=lambda x: x[1]['detection_rate'] * x[1]['avg_confidence'])
        print(f"🏆 Best performing model: {best_model[0]}")

if __name__ == "__main__":
    print("🎯 Training models with real video data")
    print("=" * 60)
    
    # Train models
    success = train_real_model()
    
    if success:
        # Test models
        test_models()
    else:
        print("❌ Training failed - cannot test models")