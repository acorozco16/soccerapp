#!/usr/bin/env python3
"""
Soccer Ball Labeling Web Interface
Click on balls to create YOLO training data
"""

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
# No Jinja2 needed - using inline HTML
from pathlib import Path
import json
import os
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Soccer Ball Labeling Interface")

# Setup paths
BASE_DIR = Path(__file__).resolve().parent
LABELING_QUEUE = BASE_DIR / "training_data" / "labeling_queue"
ANNOTATIONS_FILE = BASE_DIR / "training_data" / "annotations.json"

# Mount static files (for serving images)
app.mount("/static", StaticFiles(directory=str(LABELING_QUEUE)), name="static")

# We'll use inline HTML templates, no Jinja2 needed

class LabelingSession:
    def __init__(self):
        self.load_session()
    
    def load_session(self):
        """Load existing labeling session or create new one"""
        # Get all image files
        self.image_files = sorted([f.name for f in LABELING_QUEUE.glob("*.jpg")])
        
        # Load existing annotations
        if ANNOTATIONS_FILE.exists():
            with open(ANNOTATIONS_FILE, 'r') as f:
                self.annotations = json.load(f)
        else:
            self.annotations = {}
        
        # Find current frame
        self.current_index = 0
        for i, filename in enumerate(self.image_files):
            if filename not in self.annotations:
                self.current_index = i
                break
        else:
            # All images are labeled
            self.current_index = len(self.image_files) - 1
        
        logger.info(f"Loaded labeling session: {len(self.image_files)} total frames, starting at frame {self.current_index + 1}")
    
    def get_current_image(self) -> Optional[Dict]:
        """Get current image info"""
        if self.current_index >= len(self.image_files):
            return None
        
        filename = self.image_files[self.current_index]
        return {
            'filename': filename,
            'index': self.current_index,
            'total': len(self.image_files),
            'labeled': len(self.annotations),
            'remaining': len(self.image_files) - len(self.annotations),
            'existing_annotations': self.annotations.get(filename, [])
        }
    
    def save_annotation(self, filename: str, annotations: List[Dict]):
        """Save annotations for a frame"""
        self.annotations[filename] = annotations
        
        # Save to file
        with open(ANNOTATIONS_FILE, 'w') as f:
            json.dump(self.annotations, f, indent=2)
        
        logger.info(f"Saved {len(annotations)} annotations for {filename}")
    
    def next_frame(self):
        """Move to next unlabeled frame"""
        # Find next unlabeled frame
        for i in range(self.current_index + 1, len(self.image_files)):
            if self.image_files[i] not in self.annotations:
                self.current_index = i
                return
        
        # If no unlabeled frames found, go to next frame anyway
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
    
    def prev_frame(self):
        """Move to previous frame"""
        if self.current_index > 0:
            self.current_index -= 1
    
    def skip_frame(self):
        """Skip current frame (mark as no ball)"""
        filename = self.image_files[self.current_index]
        self.save_annotation(filename, [])  # Empty list = no ball
        self.next_frame()

# Global labeling session
session = LabelingSession()

@app.get("/", response_class=HTMLResponse)
async def labeling_interface(request: Request):
    """Main labeling interface"""
    
    current_image = session.get_current_image()
    
    if not current_image:
        return HTMLResponse("""
        <html><body>
        <h1>🎉 Labeling Complete!</h1>
        <p>All frames have been labeled.</p>
        <a href="/export">Export Dataset</a>
        </body></html>
        """)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Soccer Ball Labeling - Frame {current_image['index'] + 1}/{current_image['total']}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f0f0f0;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            .header {{
                background: #2c3e50;
                color: white;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
                text-align: center;
            }}
            .progress {{
                background: #34495e;
                height: 8px;
                border-radius: 4px;
                margin: 10px 0;
            }}
            .progress-bar {{
                background: #27ae60;
                height: 100%;
                border-radius: 4px;
                width: {(current_image['labeled'] / current_image['total']) * 100}%;
                transition: width 0.3s ease;
            }}
            .image-container {{
                position: relative;
                display: inline-block;
                border: 3px solid #34495e;
                border-radius: 8px;
                background: white;
                padding: 10px;
                margin-bottom: 20px;
            }}
            .image-container img {{
                max-width: 80vw;
                max-height: 60vh;
                cursor: crosshair;
                display: block;
            }}
            .controls {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }}
            .btn {{
                padding: 12px 24px;
                margin: 5px;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-size: 16px;
                font-weight: bold;
                transition: all 0.2s ease;
            }}
            .btn-primary {{ background: #3498db; color: white; }}
            .btn-success {{ background: #27ae60; color: white; }}
            .btn-warning {{ background: #f39c12; color: white; }}
            .btn-danger {{ background: #e74c3c; color: white; }}
            .btn-secondary {{ background: #95a5a6; color: white; }}
            .btn:hover {{ transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.2); }}
            .annotation {{
                position: absolute;
                width: 8px;
                height: 8px;
                background: red;
                border: 2px solid white;
                border-radius: 50%;
                transform: translate(-50%, -50%);
                box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            }}
            .stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 10px;
                margin-bottom: 20px;
            }}
            .stat {{
                background: white;
                padding: 15px;
                border-radius: 6px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .stat-value {{
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
            }}
            .stat-label {{
                color: #7f8c8d;
                font-size: 12px;
                text-transform: uppercase;
            }}
            .instructions {{
                background: #ecf0f1;
                padding: 15px;
                border-radius: 6px;
                margin-bottom: 20px;
                border-left: 4px solid #3498db;
            }}
            .keyboard-shortcuts {{
                background: #f8f9fa;
                padding: 10px;
                border-radius: 4px;
                font-family: monospace;
                font-size: 14px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>⚽ Soccer Ball Labeling Interface</h1>
                <div class="progress">
                    <div class="progress-bar"></div>
                </div>
                <p>Frame {current_image['index'] + 1} of {current_image['total']} | 
                   {current_image['labeled']} labeled | {current_image['remaining']} remaining</p>
            </div>
            
            <div class="stats">
                <div class="stat">
                    <div class="stat-value">{current_image['index'] + 1}</div>
                    <div class="stat-label">Current Frame</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{current_image['labeled']}</div>
                    <div class="stat-label">Completed</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{current_image['remaining']}</div>
                    <div class="stat-label">Remaining</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{len(current_image['existing_annotations'])}</div>
                    <div class="stat-label">Balls in Frame</div>
                </div>
            </div>
            
            <div class="instructions">
                <strong>Instructions:</strong> Click on each soccer ball in the image. 
                Red dots will appear where you click. If there's no ball visible, click "Skip - No Ball".
                <div class="keyboard-shortcuts">
                    <strong>Keyboard Shortcuts:</strong> Space = Save & Next | S = Skip | U = Undo | ← → = Navigate
                </div>
            </div>
            
            <div class="image-container">
                <img id="current-image" src="/static/{current_image['filename']}" alt="Frame to label" onclick="addAnnotation(event)">
                <div id="annotations"></div>
            </div>
            
            <div class="controls">
                <button class="btn btn-secondary" onclick="previousFrame()">← Previous</button>
                <button class="btn btn-warning" onclick="skipFrame()">Skip - No Ball (S)</button>
                <button class="btn btn-danger" onclick="undoLast()">Undo Last (U)</button>
                <button class="btn btn-success" onclick="saveAndNext()">Save & Next (Space)</button>
                <button class="btn btn-primary" onclick="nextFrame()">Next →</button>
            </div>
            
            <div class="controls">
                <button class="btn btn-secondary" onclick="exportDataset()">Export Dataset</button>
                <button class="btn btn-secondary" onclick="location.reload()">Refresh</button>
            </div>
        </div>
        
        <script>
            let annotations = {json.dumps(current_image['existing_annotations'])};
            let imageElement = document.getElementById('current-image');
            
            // Load existing annotations
            function loadAnnotations() {{
                const container = document.getElementById('annotations');
                container.innerHTML = '';
                
                annotations.forEach((ann, index) => {{
                    const dot = document.createElement('div');
                    dot.className = 'annotation';
                    dot.style.left = (ann.x * imageElement.clientWidth) + 'px';
                    dot.style.top = (ann.y * imageElement.clientHeight) + 'px';
                    dot.title = `Ball ${{index + 1}} (click to remove)`;
                    dot.onclick = (e) => {{
                        e.stopPropagation();
                        removeAnnotation(index);
                    }};
                    container.appendChild(dot);
                }});
                
                updateStats();
            }}
            
            function addAnnotation(event) {{
                const rect = imageElement.getBoundingClientRect();
                const x = (event.clientX - rect.left) / imageElement.clientWidth;
                const y = (event.clientY - rect.top) / imageElement.clientHeight;
                
                annotations.push({{ x: x, y: y }});
                loadAnnotations();
            }}
            
            function removeAnnotation(index) {{
                annotations.splice(index, 1);
                loadAnnotations();
            }}
            
            function undoLast() {{
                if (annotations.length > 0) {{
                    annotations.pop();
                    loadAnnotations();
                }}
            }}
            
            function updateStats() {{
                document.querySelector('.stat:last-child .stat-value').textContent = annotations.length;
            }}
            
            async function saveAndNext() {{
                await saveAnnotations();
                window.location.href = '/next';
            }}
            
            async function skipFrame() {{
                annotations = [];
                await saveAnnotations();
                window.location.href = '/next';
            }}
            
            async function saveAnnotations() {{
                const response = await fetch('/save', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{
                        filename: '{current_image['filename']}',
                        annotations: annotations
                    }})
                }});
                
                if (!response.ok) {{
                    alert('Error saving annotations!');
                }}
            }}
            
            function previousFrame() {{
                window.location.href = '/prev';
            }}
            
            function nextFrame() {{
                window.location.href = '/next';
            }}
            
            function exportDataset() {{
                window.location.href = '/export';
            }}
            
            // Keyboard shortcuts
            document.addEventListener('keydown', (e) => {{
                if (e.target.tagName === 'INPUT') return;
                
                switch(e.key) {{
                    case ' ':
                        e.preventDefault();
                        saveAndNext();
                        break;
                    case 's':
                    case 'S':
                        e.preventDefault();
                        skipFrame();
                        break;
                    case 'u':
                    case 'U':
                        e.preventDefault();
                        undoLast();
                        break;
                    case 'ArrowLeft':
                        e.preventDefault();
                        previousFrame();
                        break;
                    case 'ArrowRight':
                        e.preventDefault();
                        nextFrame();
                        break;
                }}
            }});
            
            // Load annotations on page load
            window.onload = () => {{
                loadAnnotations();
            }};
            
            // Handle image resize
            window.onresize = () => {{
                setTimeout(loadAnnotations, 100);
            }};
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

@app.post("/save")
async def save_annotations(request: Request):
    """Save annotations for current frame"""
    data = await request.json()
    filename = data['filename']
    annotations = data['annotations']
    
    # Convert relative coordinates to YOLO format if needed
    session.save_annotation(filename, annotations)
    
    return JSONResponse({"status": "saved", "count": len(annotations)})

@app.get("/next")
async def next_frame():
    """Move to next frame"""
    session.next_frame()
    return HTMLResponse('<script>window.location.href = "/";</script>')

@app.get("/prev")
async def prev_frame():
    """Move to previous frame"""
    session.prev_frame()
    return HTMLResponse('<script>window.location.href = "/";</script>')

@app.get("/export")
async def export_dataset():
    """Export annotations to YOLO format"""
    
    # Create output directory
    output_dir = BASE_DIR / "training_data" / "keepups_dataset"
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    
    for dir_path in [output_dir, images_dir, labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Copy images and create labels
    labeled_count = 0
    ball_count = 0
    
    for filename, annotations in session.annotations.items():
        # Copy image
        src_image = LABELING_QUEUE / filename
        dst_image = images_dir / filename
        
        if src_image.exists():
            import shutil
            shutil.copy2(src_image, dst_image)
            
            # Create YOLO label file
            label_file = labels_dir / f"{filename.replace('.jpg', '.txt')}"
            
            with open(label_file, 'w') as f:
                for ann in annotations:
                    # YOLO format: class_id x_center y_center width height (all normalized 0-1)
                    # For now, assume small fixed bounding box around click point
                    x_center = ann['x']
                    y_center = ann['y']
                    width = 0.05  # 5% of image width
                    height = 0.05  # 5% of image height
                    
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\\n")
                    ball_count += 1
            
            labeled_count += 1
    
    # Create dataset.yaml
    dataset_yaml = output_dir / "dataset.yaml"
    with open(dataset_yaml, 'w') as f:
        f.write(f"""path: {output_dir.absolute()}
train: images
val: images  # Using same for train/val in this case

names:
  0: ball

nc: 1
""")
    
    # Create summary
    summary = {
        'total_images': labeled_count,
        'total_annotations': ball_count,
        'output_directory': str(output_dir),
        'dataset_yaml': str(dataset_yaml)
    }
    
    summary_file = output_dir / "export_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return HTMLResponse(f"""
    <html>
    <body style="font-family: Arial; padding: 20px; background: #f0f0f0;">
        <div style="max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
            <h1 style="color: #27ae60;">🎉 Dataset Exported Successfully!</h1>
            
            <div style="background: #ecf0f1; padding: 20px; border-radius: 8px; margin: 20px 0;">
                <h3>📊 Export Summary:</h3>
                <ul style="list-style: none; padding: 0;">
                    <li><strong>Total Images:</strong> {labeled_count}</li>
                    <li><strong>Total Ball Annotations:</strong> {ball_count}</li>
                    <li><strong>Output Directory:</strong> <code>{output_dir}</code></li>
                </ul>
            </div>
            
            <div style="background: #d5f4e6; padding: 15px; border-radius: 8px; border-left: 4px solid #27ae60;">
                <h4>🚀 Next Steps:</h4>
                <ol>
                    <li>Train YOLO model with this dataset</li>
                    <li>Test improved model performance</li>
                    <li>Deploy to video processor</li>
                </ol>
            </div>
            
            <div style="text-align: center; margin-top: 30px;">
                <a href="/" style="background: #3498db; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; margin: 5px;">Continue Labeling</a>
                <button onclick="window.close()" style="background: #95a5a6; color: white; padding: 12px 24px; border: none; border-radius: 6px; margin: 5px; cursor: pointer;">Close</button>
            </div>
        </div>
    </body>
    </html>
    """)

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Soccer Ball Labeling Interface")
    print(f"📁 Labeling Queue: {LABELING_QUEUE}")
    print(f"📋 Found {len(session.image_files)} frames to label")
    print(f"✅ Already labeled: {len(session.annotations)} frames")
    print(f"🎯 Remaining: {len(session.image_files) - len(session.annotations)} frames")
    print("\n🌐 Open your browser to: http://localhost:8001")
    print("💡 Click on soccer balls in each frame to create training data")
    
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")