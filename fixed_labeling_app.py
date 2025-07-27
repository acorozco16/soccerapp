#!/usr/bin/env python3
"""
Fixed Soccer Ball Labeling Interface
"""

from fastapi import FastAPI, Request, HTTPException
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

# Ensure directories exist
LABELING_QUEUE.mkdir(parents=True, exist_ok=True)
ANNOTATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(LABELING_QUEUE)), name="static")

# Session state
session = {
    'current_index': 0,
    'annotations': {},
    'image_files': []
}

def load_session():
    """Load session data"""
    try:
        # Get image files
        session['image_files'] = sorted([f.name for f in LABELING_QUEUE.glob("*.jpg")])
        
        # Load annotations
        if ANNOTATIONS_FILE.exists():
            with open(ANNOTATIONS_FILE, 'r') as f:
                session['annotations'] = json.load(f)
        else:
            session['annotations'] = {}
            
        # Find first unlabeled image
        session['current_index'] = 0
        for i, filename in enumerate(session['image_files']):
            if filename not in session['annotations']:
                session['current_index'] = i
                break
                
        logger.info(f"Session loaded: {len(session['image_files'])} images, {len(session['annotations'])} labeled")
        
    except Exception as e:
        logger.error(f"Error loading session: {e}")
        session['image_files'] = []
        session['annotations'] = {}
        session['current_index'] = 0

def save_annotations():
    """Save annotations to file"""
    try:
        with open(ANNOTATIONS_FILE, 'w') as f:
            json.dump(session['annotations'], f, indent=2)
        logger.info(f"Annotations saved: {len(session['annotations'])} files")
    except Exception as e:
        logger.error(f"Error saving annotations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save annotations: {e}")

# Load session on startup
load_session()

@app.get("/", response_class=HTMLResponse)
async def home():
    try:
        if not session['image_files']:
            return HTMLResponse("<h1>No images found to label!</h1>")
            
        if session['current_index'] >= len(session['image_files']):
            return HTMLResponse(f"""
            <h1>🎉 All Done!</h1>
            <p>Labeled {len(session['annotations'])} out of {len(session['image_files'])} images.</p>
            <a href="/export">Export Dataset</a>
            """)
        
        filename = session['image_files'][session['current_index']]
        existing_balls = session['annotations'].get(filename, [])
        
        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Ball Labeling - {session['current_index'] + 1}/{len(session['image_files'])}</title>
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
                    <p>Frame {session['current_index'] + 1} of {len(session['image_files'])}</p>
                    <p>Progress: {len(session['annotations'])}/{len(session['image_files'])} labeled</p>
                </div>
                
                <div class="stats">
                    <strong>Instructions:</strong> Click on soccer balls. Red dots = your clicks. 
                    Current frame has <span id="ball-count">{len(existing_balls)}</span> ball(s) marked.
                </div>
                
                <div class="image-container">
                    <img id="image" src="/static/{filename}" onclick="addBall(event)" onload="updateDisplay()">
                    <div id="balls"></div>
                </div>
                
                <div class="controls">
                    <button class="btn btn-warning" onclick="skipFrame()">Skip - No Ball</button>
                    <button class="btn btn-danger" onclick="undoLast()">Undo Last</button>
                    <button class="btn btn-success" onclick="saveAndNext()">Save & Next</button>
                </div>
            </div>
            
            <script>
                let balls = {json.dumps(existing_balls)};
                let img = document.getElementById('image');
                
                function addBall(event) {{
                    try {{
                        let rect = img.getBoundingClientRect();
                        let x = (event.clientX - rect.left) / img.clientWidth;
                        let y = (event.clientY - rect.top) / img.clientHeight;
                        
                        balls.push({{x: x, y: y}});
                        updateDisplay();
                    }} catch(e) {{
                        console.error('Error adding ball:', e);
                    }}
                }}
                
                function updateDisplay() {{
                    try {{
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
                        
                        document.getElementById('ball-count').textContent = balls.length;
                    }} catch(e) {{
                        console.error('Error updating display:', e);
                    }}
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
                    try {{
                        await saveBalls();
                        window.location.href = '/next';
                    }} catch(e) {{
                        alert('Error saving: ' + e.message);
                    }}
                }}
                
                async function skipFrame() {{
                    try {{
                        balls = [];
                        await saveBalls();
                        window.location.href = '/next';
                    }} catch(e) {{
                        alert('Error saving: ' + e.message);
                    }}
                }}
                
                async function saveBalls() {{
                    const response = await fetch('/save', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{
                            filename: '{filename}',
                            balls: balls
                        }})
                    }});
                    
                    if (!response.ok) {{
                        const error = await response.text();
                        throw new Error(error);
                    }}
                }}
            </script>
        </body>
        </html>
        """)
        
    except Exception as e:
        logger.error(f"Error in home route: {e}")
        return HTMLResponse(f"<h1>Error: {e}</h1>")

@app.post("/save")
async def save_balls(request: Request):
    try:
        data = await request.json()
        filename = data['filename']
        balls = data['balls']
        
        # Validate data
        if not isinstance(filename, str) or not isinstance(balls, list):
            raise HTTPException(status_code=400, detail="Invalid data format")
        
        # Save to session
        session['annotations'][filename] = balls
        
        # Save to file
        save_annotations()
        
        logger.info(f"Saved {len(balls)} balls for {filename}")
        return JSONResponse({"status": "saved", "count": len(balls)})
        
    except Exception as e:
        logger.error(f"Error saving balls: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/next")
async def next_frame():
    try:
        session['current_index'] += 1
        return HTMLResponse('<script>window.location.href = "/";</script>')
    except Exception as e:
        logger.error(f"Error in next route: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/export")
async def export_dataset():
    try:
        # Create dataset
        output_dir = BASE_DIR / "training_data" / "keepups_dataset"
        images_dir = output_dir / "images"
        labels_dir = output_dir / "labels"
        
        for dir_path in [output_dir, images_dir, labels_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        import shutil
        total_images = 0
        total_balls = 0
        
        for filename, balls in session['annotations'].items():
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
        
    except Exception as e:
        logger.error(f"Error exporting dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    print(f"🚀 Starting fixed labeling interface...")
    print(f"📸 Found {len(session['image_files'])} images")
    print(f"✅ Already labeled: {len(session['annotations'])}")
    print(f"🌐 Open: http://localhost:8002")
    
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="warning")