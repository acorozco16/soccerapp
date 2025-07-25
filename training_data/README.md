# Soccer Training Data Collection System

A comprehensive system for collecting, processing, and training soccer ball detection models using YouTube videos and YOLOv8.

## 🎯 Overview

This system automatically:
1. **Scrapes soccer training videos** from YouTube with rate limiting
2. **Processes frames** and creates bounding box labels automatically
3. **Manages datasets** with quality categorization and version control
4. **Trains YOLOv8 models** with automated hyperparameter tuning
5. **Evaluates and deploys** the best performing models

## 📁 Directory Structure

```
training_data/
├── scrapers/           # YouTube video scraping
│   └── youtube_scraper.py
├── processors/         # Frame processing and labeling
│   └── frame_processor.py
├── datasets/          # Dataset management and YOLO export
│   └── dataset_manager.py
├── models/            # YOLOv8 training and evaluation
│   └── yolo_trainer.py
├── scripts/           # Main pipeline orchestration
│   └── run_training_pipeline.py
├── scraped_data/      # Raw scraped videos and frames
├── processed_dataset/ # Processed frames with annotations
├── datasets/          # YOLO format datasets
├── models/            # Trained models
├── experiments/       # Training experiments
└── README.md
```

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
cd soccer-app/backend
pip install -r requirements.txt

# Additional dependencies for training
pip install scikit-learn pandas matplotlib seaborn
```

### 1. Quick Test Run

```bash
cd training_data/scripts
python run_training_pipeline.py --quick-test
```

This will:
- Download 5 videos
- Process 50 frames
- Create a dataset with 20 images
- Train for 5 epochs

### 2. Full Pipeline Run

```bash
python run_training_pipeline.py \
    --max-videos 100 \
    --max-images 1000 \
    --model-size yolov8s \
    --epochs 100
```

### 3. Individual Components

#### YouTube Scraping Only
```bash
cd scrapers
python youtube_scraper.py
```

#### Frame Processing Only
```bash
cd processors
python frame_processor.py
```

#### Dataset Creation Only
```bash
cd datasets
python dataset_manager.py
```

#### Model Training Only
```bash
cd models
python yolo_trainer.py
```

## 🔧 Configuration Options

### Data Collection
- `--max-videos`: Maximum videos to download (default: 50)
- `--max-videos-per-term`: Videos per search term (default: 10)
- `--quality-filter`: Quality levels to include (default: high, medium)

### Training
- `--model-size`: YOLOv8 model size (yolov8n/s/m, default: yolov8s)
- `--epochs`: Training epochs (default: 50)
- `--max-images`: Maximum images in dataset (default: 500)

### Pipeline Control
- `--collect-data`: Run data collection phase
- `--process-frames`: Run frame processing phase
- `--create-dataset`: Run dataset creation phase
- `--train-model`: Run model training phase
- `--evaluate`: Run model evaluation
- `--deploy`: Deploy best model

## 📊 Features

### 1. YouTube Video Scraper
- **Rate Limited**: 20 requests/minute to prevent API abuse
- **Quality Filtering**: Only videos 30sec-5min with good ball visibility
- **Duplicate Detection**: Avoids re-downloading existing videos
- **Error Handling**: Robust error recovery and logging
- **Progress Tracking**: Detailed progress and statistics

**Search Terms Used:**
- "soccer ball juggling training"
- "football skills practice"
- "soccer dribbling drills"
- "football ball control"
- "soccer first touch training"
- And more...

### 2. Frame Processing & Auto-Labeling
- **Multi-Method Ball Detection**:
  - HSV color filtering (orange/white balls)
  - Hough circle transform
  - Contour detection for irregular shapes
- **Quality Assessment**: Brightness, contrast, blur analysis
- **Duplicate Removal**: Hash-based duplicate detection
- **YOLO Format**: Automatic bounding box annotation
- **Categorization**: High/medium/low quality classification

### 3. Dataset Management
- **Version Control**: Track dataset versions with metadata
- **Balanced Splits**: Stratified train/val/test splits
- **Diversity Metrics**: Calculate dataset diversity scores
- **YOLO Export**: Direct export to YOLOv8 training format
- **Quality Filtering**: Filter by image quality categories
- **Statistics**: Comprehensive dataset analysis

### 4. YOLOv8 Training Pipeline
- **Multiple Model Sizes**: Support for YOLOv8n/s/m variants
- **Automated Training**: Hyperparameter optimization
- **Progress Monitoring**: Real-time training visualization
- **Early Stopping**: Prevent overfitting with patience
- **Model Comparison**: Compare different model versions
- **Deployment**: Automatic deployment of best models

## 📈 Monitoring & Evaluation

### Training Metrics
- Precision, Recall, mAP@0.5, mAP@0.5:0.95
- Training curves and loss visualization
- Learning rate scheduling
- Validation metrics tracking

### Dataset Quality Metrics
- Image quality distribution
- Lighting condition analysis
- Ball visibility assessment
- Diversity scoring
- Bounding box statistics

### Model Performance
- Before/after comparison
- Test set evaluation
- Inference speed benchmarks
- Memory usage analysis

## 🔍 Quality Assurance

### Data Quality
- **Multi-layer validation**: Physics-based ball detection
- **Quality scoring**: Automatic image quality assessment
- **Duplicate prevention**: Hash-based duplicate detection
- **Manual review flags**: Automatic flagging of questionable data

### Training Quality
- **Cross-validation**: Stratified dataset splits
- **Hyperparameter tuning**: Automated parameter optimization
- **Early stopping**: Prevent overfitting
- **Model versioning**: Track all training experiments

## 📝 Output Files

### Scraped Data
```
scraped_data/
├── videos/                    # Downloaded MP4 videos
├── frames/                    # Extracted frames (0.5s intervals)
└── metadata/                  # Video and frame metadata
    ├── downloaded.json        # List of downloaded videos
    └── scraping_summary_*.json # Scraping session summaries
```

### Processed Dataset
```
processed_dataset/
├── images/                    # Organized by quality
│   ├── high_quality/
│   ├── medium_quality/
│   └── low_quality/
├── labels/                    # YOLO format labels
├── metadata/                  # Processing metadata
└── visualizations/            # Debug visualizations
```

### YOLO Datasets
```
datasets/yolo_v{version}/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
├── dataset.yaml              # YOLO config file
└── metadata.json             # Dataset metadata
```

### Trained Models
```
models/
├── soccer_ball_{model}_{version}.pt  # Best trained models
├── training_history.json             # All training runs
└── model_comparison_*.json           # Model comparisons
```

## 🛠 Troubleshooting

### Common Issues

**1. YouTube Download Failures**
```bash
# Check rate limiting
# Verify yt-dlp is updated
pip install -U yt-dlp
```

**2. CUDA Out of Memory**
```bash
# Reduce batch size
python run_training_pipeline.py --model-size yolov8n --epochs 50
```

**3. Poor Ball Detection**
```bash
# Use only high quality images
python run_training_pipeline.py --quality-filter high
```

**4. Low Dataset Diversity**
```bash
# Increase max videos and check diversity metrics
python run_training_pipeline.py --max-videos 200
```

### Logs and Debugging
- Check `training_pipeline.log` for detailed logs
- Monitor GPU usage with `nvidia-smi`
- Review dataset statistics in metadata files
- Check training curves in experiment directories

## 📊 Expected Results

### Performance Targets
- **Data Collection**: 50-100 videos/hour (rate limited)
- **Frame Processing**: 100-500 frames/minute
- **Training Time**: 2-6 hours for 100 epochs (depends on GPU)
- **Model Accuracy**: mAP@0.5 > 0.85 for good datasets

### Dataset Quality
- **Ball Visibility**: >70% frames with clear ball detection
- **Quality Distribution**: 40% high, 40% medium, 20% low quality
- **Diversity Score**: >0.7 for balanced lighting/conditions

## 🔄 Integration with Main App

The trained models can be integrated back into the main soccer analysis app:

```python
# In video_processor.py
from ultralytics import YOLO

# Load trained model
model = YOLO('./training_data/deployed_model/best_soccer_ball_model.pt')

# Use in detection pipeline
results = model(frame, conf=0.5)
```

## 📜 License

Same as main project - use responsibly and respect YouTube's terms of service.

## 🤝 Contributing

1. Add new search terms in `youtube_scraper.py`
2. Improve ball detection algorithms in `frame_processor.py`
3. Add new model architectures in `yolo_trainer.py`
4. Enhance quality metrics in `dataset_manager.py`

---

**Note**: This system is designed for research and educational purposes. Always respect YouTube's terms of service and rate limits when scraping content.