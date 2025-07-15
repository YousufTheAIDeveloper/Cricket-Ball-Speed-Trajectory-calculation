import cv2
import numpy as np
import torch
from pathlib import Path

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

#In the COCO Dataset, these are the class indices for a player and ball
BALL_CLASS = 32  
PLAYER_CLASS = 0 

input_video = 'cricket_video.mp4'  # Add the path to your video
output_dir = Path('results')
output_dir.mkdir(exist_ok=True)
output_video = str(output_dir / 'cricket_detected.mp4')

cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

ball_positions = []
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO inference
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()

    # Draw detections and track ball
    ball_center = None
    for *xyxy, conf, cls in detections:
        cls = int(cls)
        if cls == BALL_CLASS and conf > 0.3:
            x1, y1, x2, y2 = map(int, xyxy)
            ball_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, 'Ball', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        elif cls == PLAYER_CLASS and conf > 0.3:
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, 'Player', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Track ball positions
    if ball_center:
        ball_positions.append((frame_count, ball_center))

    # Draw trajectory
    for i in range(1, len(ball_positions)):
        pt1 = ball_positions[i - 1][1]
        pt2 = ball_positions[i][1]
        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    # Calculate speed (pixels/frame, can be converted to real units if calibration is known)
    if len(ball_positions) >= 2:
        f1, p1 = ball_positions[-2]
        f2, p2 = ball_positions[-1]
        dist = np.linalg.norm(np.array(p2) - np.array(p1))
        
        feet_per_pixel = 0.22  # Example, based on calibration
        dist_pixels = np.linalg.norm(np.array(p2) - np.array(p1))
        speed_fps = dist_pixels * feet_per_pixel * fps  # feet per second
        speed_mph = speed_fps * 0.681818  # 1 ft/s = 0.681818 mph
        
        cv2.putText(frame, f'Speed: {speed_mph:.2f} mph', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # speed = dist * fps  # pixels/second
        # cv2.putText(frame, f'Speed: {speed:.2f} px/s', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    out.write(frame)
    frame_count += 1

cap.release()
out.release()
print(f"Processed video saved to {output_video}")