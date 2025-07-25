# Soccer Ball Touch Counter - Video Analysis App

A web application that analyzes soccer videos to automatically count ball touches using computer vision. Perfect for parents and coaches to track player performance.

## 🎯 Features

- Upload soccer videos from your phone (MP4/MOV format)
- Automatic ball detection and player tracking
- Counts ball touches with confidence scoring
- Shows annotated frames as proof
- Mobile-friendly interface
- Process videos in under 3 minutes

## 🚀 Quick Start (For Non-Technical Users)

### Prerequisites

You'll need to install these programs first:

1. **Python 3.9 or higher**
   - Mac: Open Terminal and run `brew install python3`
   - Windows: Download from [python.org](https://python.org)

2. **Node.js 18 or higher**
   - Download from [nodejs.org](https://nodejs.org)

3. **Git** (for version control)
   - Mac: Comes pre-installed
   - Windows: Download from [git-scm.com](https://git-scm.com)

### Step 1: Download the Code

1. Open Terminal (Mac) or Command Prompt (Windows)
2. Navigate to where you want to install:
   ```bash
   cd ~/Desktop
   ```
3. Clone the repository:
   ```bash
   git clone [your-repository-url]
   cd soccer-app
   ```

### Step 2: Set Up the Backend (Video Processing)

1. Navigate to backend folder:
   ```bash
   cd backend
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```

3. Activate it:
   - Mac/Linux: `source venv/bin/activate`
   - Windows: `venv\Scripts\activate`

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Start the backend server:
   ```bash
   python main.py
   ```
   
   You should see: `Server running at http://localhost:8000`

### Step 3: Set Up the Frontend (Web Interface)

1. Open a NEW terminal window
2. Navigate to frontend folder:
   ```bash
   cd soccer-app/frontend
   ```

3. Install dependencies:
   ```bash
   npm install
   ```

4. Start the frontend:
   ```bash
   npm run dev
   ```
   
   You should see: `Local: http://localhost:3000`

### Step 4: Use the App!

1. Open your web browser
2. Go to: http://localhost:3000
3. Click "Upload Video" and select a soccer video from your phone
4. Wait for processing (usually 1-3 minutes)
5. View your results!

## 📱 Recording Tips for Best Results

- Record in landscape mode
- Keep the camera steady
- Ensure good lighting
- Try to keep the ball in frame
- 30-60 seconds is ideal length
- Avoid zooming in/out while recording

## 🧪 Testing the App

We've included sample videos to test with:

```bash
cd backend
python analyze_sample.py --video ../sample_videos/clear_touches.mp4
```

## 🚨 Troubleshooting

### "Command not found" errors
- Make sure Python and Node.js are installed
- Restart your terminal after installation

### "Module not found" errors
- Make sure you activated the virtual environment
- Re-run `pip install -r requirements.txt`

### Video processing takes too long
- Try shorter videos (30-60 seconds)
- Ensure your computer isn't running other heavy programs
- Use "Quick Mode" if available

### No ball detected
- Ensure the ball is clearly visible
- Try better lighting conditions
- The ball should be orange or white

## 🌐 Deployment (Making it Live)

### Frontend (Vercel)
1. Create account at [vercel.com](https://vercel.com)
2. Connect your GitHub repository
3. Deploy with one click!

### Backend (Railway)
1. Create account at [railway.app](https://railway.app)
2. Connect your GitHub repository
3. Add environment variables from `.env.example`
4. Deploy!

## 📊 Understanding Results

- **Total Touches**: Number of times the ball contacted the player
- **Confidence Score**: How sure the AI is (0.8+ is very confident)
- **Touches per Minute**: Normalized for video length
- **Debug Frames**: Visual proof of detected touches

## 🛟 Getting Help

- Check the sample videos first
- Review error messages carefully
- Ensure all steps were followed in order
- Contact: [your-email@example.com]

## 📝 License

MIT License - feel free to use for your team!