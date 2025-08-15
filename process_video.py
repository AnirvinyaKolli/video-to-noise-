import cv2
import numpy as np
from pathlib import Path
import os

documents_dir = Path(os.path.expanduser("~/Documents"))



detection_scale = 125

def sanitize_frame(f, target_size):
    if f.ndim == 2:
        f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
    elif f.shape[2] == 4:
        f = cv2.cvtColor(f, cv2.COLOR_BGRA2BGR)
    if f.dtype != np.uint8:
        f = f.astype(np.uint8)
    if f.shape[:2] != target_size:
        f = cv2.resize(f, (target_size[1], target_size[0]))
    return f

def compare_frames(frame1, frame2, prev_bw_frame):

    diff = cv2.absdiff(frame1, frame2)
    new_bw_frame = prev_bw_frame.copy()
    mask = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) < detection_scale
    coords = np.where(mask)
    for y, x in zip(*coords):
        new_bw_frame[y, x] = 255 - prev_bw_frame[y, x]
    return new_bw_frame

prev_processed_frame = None
prev_frame = None
height = 0
width = 0

cap = cv2.VideoCapture('test_video.mp4')

fourcc = cv2.VideoWriter.fourcc(*'XVID')
output_dir = documents_dir / "python_video_processing"
output_dir.mkdir(parents=True, exist_ok=True)
filepath = output_dir / "output.avi"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if prev_frame is None:
        height, width = frame.shape[:2]
        out = cv2.VideoWriter(str(filepath), fourcc, 20.0, (width, height))
        bw_random = np.random.choice([0, 255], size=(height, width), p=[0.5, 0.5]).astype(np.uint8)
        bw_frame = cv2.merge([bw_random, bw_random, bw_random])
        bw_frame = sanitize_frame(bw_frame, (height, width))
        out.write(bw_frame)
        prev_processed_frame = bw_frame
    else:
        new_bw_frame = compare_frames(prev_frame, frame, prev_processed_frame)
        new_bw_frame = sanitize_frame(new_bw_frame, (height, width))
        out.write(new_bw_frame)
        prev_processed_frame = new_bw_frame

    prev_frame = frame

out.release()
cap.release()
cv2.destroyAllWindows()




