import cv2
import numpy as np

detection_scale = 125


def compare_frames(frame1, frame2, prev_bw_frame):

    diff = cv2.absdiff(frame1, frame2)
    new_bw_frame = prev_bw_frame.copy()
    mask = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) < detection_scale
    coords = np.where(mask)
    for y, x in zip(*coords):
        new_bw_frame[y, x] = 255 - prev_bw_frame[y, x]
    return new_bw_frame

cap = cv2.VideoCapture('test_video.mp4')

prev_frame = None
height = 0
width = 0
new_frames = []



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if prev_frame is None:
        height, width = frame.shape[:2]
        bw_random = np.random.choice([0, 255], size=(height, width), p=[0.5, 0.5]).astype(np.uint8)
        bw_frame = cv2.merge([bw_random, bw_random, bw_random])
        new_frames.append(bw_frame)
    else:
        new_bw_frame = compare_frames(prev_frame, frame, new_frames[len(new_frames)-1])
        new_frames.append(new_bw_frame)

    prev_frame = frame

cap.release()
cv2.destroyAllWindows()


fourcc = cv2.VideoWriter.fourcc(*'XVID')
filepath = 'C:/Users/anirv/Documents/python_video_processing/out.avi'
out = cv2.VideoWriter(filepath, fourcc, 20.0, (width, height))



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

for frame in new_frames:
    frame = sanitize_frame(frame, (height, width))
    out.write(frame)


out.release()
