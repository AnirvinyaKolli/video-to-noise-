from flask import Flask, request, send_file, render_template
import cv2
import numpy as np
import tempfile

app = Flask(__name__)

def process_video(input_path, output_path, detection_scale):

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
        mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) > detection_scale
        coords = np.where(mask)
        for y, x in zip(*coords):
            new_bw_frame[y, x] = 255 - prev_bw_frame[y, x]
        return new_bw_frame

    cap = cv2.VideoCapture(input_path)
    prev_processed_frame = None
    prev_frame = None

    ret, frame = cap.read()
    if not ret:
        cap.release()
        return

    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter.fourcc(*'XVID')
    out = cv2.VideoWriter(str(output_path), fourcc, 20.0, (width, height))

    bw_random = np.random.choice([0, 255], size=(height, width), p=[0.5, 0.5]).astype(np.uint8)
    bw_frame = cv2.merge([bw_random, bw_random, bw_random])
    bw_frame = sanitize_frame(bw_frame, (height, width))
    out.write(bw_frame)
    prev_processed_frame = bw_frame
    prev_frame = frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        new_bw_frame = compare_frames(prev_frame, frame, prev_processed_frame)
        new_bw_frame = sanitize_frame(new_bw_frame, (height, width))
        out.write(new_bw_frame)
        prev_processed_frame = new_bw_frame
        prev_frame = frame

    out.release()
    cap.release()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded"
        file = request.files['file']
        detection_scale = int(request.form['detection_scale'])
        if file.filename == '':
            return "No file selected"

        with tempfile.NamedTemporaryFile(delete=False, suffix=".avi") as temp_input:
            file.save(temp_input.name)
            temp_input_path = temp_input.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".avi") as temp_output:
            temp_output_path = temp_output.name

        process_video(temp_input_path, temp_output_path, detection_scale)

        return send_file(temp_output_path, as_attachment=True, download_name=f"processed_{file.filename}")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

#process_video("C:\\Users\\anirv\\Downloads\\test_video.mp4", 'C:\\Users\\anirv\\Downloads\\output_video.avi')