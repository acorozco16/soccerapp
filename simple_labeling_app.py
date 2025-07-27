#!/usr/bin/env python3
"""
Simple Soccer Ball Labeling Interface
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Soccer Ball Labeling")

# Setup paths
BASE_DIR = Path(__file__).resolve().parent
LABELING_QUEUE = BASE_DIR / "training_data" / "labeling_queue"
ANNOTATIONS_FILE = BASE_DIR / "training_data" / "annotations.json"

# Mount static files
app.mount("/static", StaticFiles(directory=str(LABELING_QUEUE)), name="static")

# Global session state
class Session:
    def __init__(self):
        self.image_files = sorted([f.name for f in LABELING_QUEUE.glob("*.jpg")])
        self.current_index = 0
        self.annotations = {}
        self.load_annotations()
    
    def load_annotations(self):
        if ANNOTATIONS_FILE.exists():
            with open(ANNOTATIONS_FILE, 'r') as f:
                self.annotations = json.load(f)
    
    def save_annotations(self):
        with open(ANNOTATIONS_FILE, 'w') as f:
            json.dump(self.annotations, f, indent=2)
    
    def get_current_image(self):
        if self.current_index >= len(self.image_files):
            return None
        filename = self.image_files[self.current_index]
        return {
            'filename': filename,
            'index': self.current_index,
            'total': len(self.image_files),
            'existing': self.annotations.get(filename, [])
        }

session = Session()

@app.get("/", response_class=HTMLResponse)
async def home():
    current = session.get_current_image()
    
    if not current:
        return HTMLResponse("<h1>All done! No more images to label.</h1>")
    
    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Soccer Ball Labeling - Frame {current['index'] + 1}/{current['total']}</title>
        <style>
            body {{ font-family: Arial; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }}
            .header {{ background: #2c3e50; color: white; padding: 15px; border-radius: 5px; text-align: center; margin-bottom: 20px; }}
            .image-container {{ position: relative; border: 2px solid #333; margin: 20px 0; display: inline-block; }}
            .image-container img {{ max-width: 800px; max-height: 600px; cursor: crosshair; }}
            .controls {{ margin: 20px 0; }}
            .btn {{ padding: 10px 20px; margin: 5px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }}
            .btn-primary {{ background: #3498db; color: white; }}
            .btn-success {{ background: #27ae60; color: white; }}
            .btn-warning {{ background: #f39c12; color: white; }}
            .btn-danger {{ background: #e74c3c; color: white; }}
            .annotation {{ position: absolute; width: 6px; height: 6px; background: red; border: 2px solid white; border-radius: 50%; transform: translate(-50%, -50%); }}
            .stats {{ background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>⚽ Soccer Ball Labeling</h1>
                <p>Frame {current['index'] + 1} of {current['total']}</p>
                <p>Click on each soccer ball in the image</p>
            </div>
            
            <div class="stats">
                <strong>Instructions:</strong> Click on soccer balls to mark their positions. 
                If no ball is visible, click "Skip - No Ball".
                Current frame has {len(current['existing'])} ball(s) marked.
            </div>
            
            <div class="image-container">
                <img id="image" src="/static/{current['filename']}" onclick="addBall(event)">
                <div id="balls"></div>
            </div>
            
            <div class="controls">
                <button class="btn btn-warning" onclick="skipFrame()">Skip - No Ball</button>
                <button class="btn btn-danger" onclick="undoLast()">Undo Last</button>
                <button class="btn btn-success" onclick="saveAndNext()">Save & Next</button>
                <button class="btn btn-primary" onclick="nextFrame()">Next Frame</button>
            </div>
            
            <div class="controls">
                <button class="btn btn-primary" onclick="exportDataset()">Export Dataset</button>
            </div>
        </div>
        
        <script>
            let balls = {json.dumps(current['existing'])};
            let img = document.getElementById('image');
            
            function addBall(event) {{
                let rect = img.getBoundingClientRect();
                let x = (event.clientX - rect.left) / img.clientWidth;
                let y = (event.clientY - rect.top) / img.clientHeight;
                
                balls.push({{x: x, y: y}});
                updateDisplay();
            }}
            
            function updateDisplay() {{
                let container = document.getElementById('balls');
                container.innerHTML = '';
                
                balls.forEach((ball, i) => {{
                    let dot = document.createElement('div');
                    dot.className = 'annotation';
                    dot.style.left = (ball.x * img.clientWidth) + 'px';
                    dot.style.top = (ball.y * img.clientHeight) + 'px';
                    dot.onclick = (e) => {{ e.stopPropagation(); removeBall(i); }};
                    container.appendChild(dot);
                }});
            }}
            
            function removeBall(index) {{
                balls.splice(index, 1);
                updateDisplay();
            }}
            
            function undoLast() {{
                if (balls.length > 0) {{
                    balls.pop();
                    updateDisplay();
                }}
            }}
            
            async function saveAndNext() {{
                await saveBalls();
                window.location.href = '/next';
            }}
            
            async function skipFrame() {{
                balls = [];
                await saveBalls();
                window.location.href = '/next';
            }}
            
            function nextFrame() {{
                window.location.href = '/next';
            }}
            
            function exportDataset() {{
                window.location.href = '/export';
            }}
            
            async function saveBalls() {{
                await fetch('/save', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{
                        filename: '{current['filename']}',
                        balls: balls
                    }})
                }});
            }}
            
            // Load existing balls
            window.onload = updateDisplay;
            window.onresize = updateDisplay;
        </script>
    </body>
    </html>
    """)

@app.post("/save")
async def save_balls(request: Request):
    data = await request.json()
    filename = data['filename']
    balls = data['balls']
    
    session.annotations[filename] = balls
    session.save_annotations()
    
    return JSONResponse({"status": "saved"})

@app.get("/next")
async def next_frame():
    session.current_index += 1
    return HTMLResponse('<script>window.location.href = "/";</script>')

@app.get("/export")
async def export_dataset():
    # Create dataset
    output_dir = BASE_DIR / "training_data" / "keepups_dataset"
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    
    for dir_path in [output_dir, images_dir, labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    import shutil
    total_images = 0
    total_balls = 0
    
    for filename, balls in session.annotations.items():
        # Copy image
        src = LABELING_QUEUE / filename
        dst = images_dir / filename
        
        if src.exists():
            shutil.copy2(src, dst)
            
            # Create label file
            label_file = labels_dir / filename.replace('.jpg', '.txt')
            with open(label_file, 'w') as f:
                for ball in balls:
                    # YOLO format: class x_center y_center width height
                    x_center = ball['x']
                    y_center = ball['y']
                    width = 0.05  # 5% of image
                    height = 0.05
                    
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\\n")
                    total_balls += 1
            
            total_images += 1
    
    # Create dataset.yaml
    with open(output_dir / "dataset.yaml", 'w') as f:
        f.write(f"""path: {output_dir.absolute()}
train: images
val: images

names:
  0: ball

nc: 1
""")
    
    return HTMLResponse(f"""
    <html>
    <body style="font-family: Arial; padding: 20px;">
        <h1>🎉 Dataset Exported!</h1>
        <p><strong>Images:</strong> {total_images}</p>
        <p><strong>Ball annotations:</strong> {total_balls}</p>
        <p><strong>Location:</strong> {output_dir}</p>
        
        <h3>Next Steps:</h3>
        <ol>
            <li>Train YOLO model with this dataset</li>
            <li>Test improved performance</li>
        </ol>
        
        <a href="/" style="background: #3498db; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Continue Labeling</a>
    </body>
    </html>
    """)

if __name__ == "__main__":
    import uvicorn
    
    print(f"🚀 Starting labeling interface...")
    print(f"📸 Found {len(session.image_files)} images")
    print(f"✅ Already labeled: {len(session.annotations)}")
    print(f"🌐 Open: http://localhost:8001")
    
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="warning")