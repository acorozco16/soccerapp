from ultralytics import YOLO
import cv2

model = YOLO('training_data/experiments/real_detector_v2/weights/best.pt')
video = cv2.VideoCapture('uploads/raw/20250726_125300_5710ec27.mp4')

print('Testing NEW video...')
for i in range(5):
    ret, frame = video.read()
    if ret:
        results = model(frame, conf=0.01, verbose=False)
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                max_conf = max(float(b.conf[0]) for b in r.boxes)
                print(f'Frame {i}: {len(r.boxes)} detections, max conf: {max_conf:.4f}')
            else:
                print(f'Frame {i}: No detections')
    else:
        print(f'Frame {i}: Could not read frame')

video.release()
print('Done testing')