"""
Real-Time Automated Exam Invigilation System
Main Web Application Server & Video Stream Orchestrator
Author: Venkatrao Yasarapu
"""

import os
import sys
import time
import queue
import threading
import cv2
from flask import Flask, render_template, Response, jsonify, request

from invigilator.head_pose import HeadPoseEstimator
from invigilator.face_recognizer import FaceVerifier
from invigilator.object_detector import ObjectDetector

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

app = Flask(__name__)

# Global streaming state
camera = None
camera_lock = threading.Lock()
is_monitoring = False
event_logs = []
alert_counter = 0
last_beep_time = 0

# Initialize modules
print("[System] Initializing AI Invigilation Modules...")
pose_estimator = HeadPoseEstimator(landmark_model_path="shape_predictor_68_face_landmarks.dat")
face_verifier = FaceVerifier(known_faces_dir="known_faces")
object_detector = ObjectDetector(
    config_path="models/yolo-coco/yolov3.cfg",
    weights_path="models/yolo-coco/yolov3.weights",
    labels_path="models/yolo-coco/coco.names"
)
print("[System] All modules loaded.")


def play_audio_alert():
    """Triggers an audio beep in a background thread without blocking video frames."""
    global last_beep_time
    now = time.time()
    if now - last_beep_time < 1.2:
        return
    last_beep_time = now

    def _beep():
        if HAS_WINSOUND:
            try:
                winsound.Beep(850, 450)
            except Exception:
                pass
    threading.Thread(target=_beep, daemon=True).start()


def get_camera():
    """Safely retrieves or opens the webcam capture."""
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
    return camera


def release_camera():
    """Safely releases the webcam capture."""
    global camera
    with camera_lock:
        if camera is not None and camera.isOpened():
            camera.release()
        camera = None


def generate_frames():
    """Generator yielding multipart MJPEG frames processed by the CV pipeline."""
    global is_monitoring, alert_counter, event_logs

    cap = get_camera()
    fps_time = time.time()
    frame_count = 0
    fps = 0.0

    while is_monitoring:
        success, frame = cap.read()
        if not success:
            time.sleep(0.05)
            continue

        frame_count += 1
        current_time = time.time()
        if current_time - fps_time >= 1.0:
            fps = frame_count / (current_time - fps_time)
            frame_count = 0
            fps_time = current_time

        frame_violations = []

        # 1. Head Pose & Facial Movement
        frame, pose_data, pose_alerts = pose_estimator.process_frame(frame)
        frame_violations.extend(pose_alerts)

        # 2. Candidate Face Verification
        frame, names, id_alerts = face_verifier.verify_frame(frame)
        frame_violations.extend(id_alerts)

        # 3. Prohibited Object Detection
        frame, detected_objs, obj_alerts = object_detector.detect_objects(frame)
        frame_violations.extend(obj_alerts)

        # Process Violations
        if frame_violations:
            play_audio_alert()
            for violation in frame_violations:
                timestamp = time.strftime("%H:%M:%S")
                log_entry = {
                    "time": timestamp,
                    "text": violation,
                    "type": "danger"
                }
                # Avoid flooding identical logs within 1 second
                if not event_logs or event_logs[-1]["text"] != violation or event_logs[-1]["time"] != timestamp:
                    event_logs.append(log_entry)
                    alert_counter += 1
                    if len(event_logs) > 200:
                        event_logs.pop(0)

        # Draw HUD overlays
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 110, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Watermark
        cv2.putText(frame, "AI INVIGILATOR ACTIVE", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    """Renders the proctoring dashboard interface."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    global is_monitoring
    if not is_monitoring:
        is_monitoring = True
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/start', methods=['POST'])
def start_monitoring():
    """Starts candidate monitoring."""
    global is_monitoring
    is_monitoring = True
    get_camera()
    return jsonify({"status": "started", "monitoring": True})


@app.route('/api/stop', methods=['POST'])
def stop_monitoring():
    """Stops candidate monitoring and releases the camera."""
    global is_monitoring
    is_monitoring = False
    release_camera()
    return jsonify({"status": "stopped", "monitoring": False})


@app.route('/api/status')
def get_status():
    """Returns telemetry and statistics."""
    global is_monitoring, alert_counter
    return jsonify({
        "monitoring": is_monitoring,
        "total_alerts": alert_counter,
        "logs_count": len(event_logs),
        "yolo_ready": object_detector.ready,
        "dlib_ready": pose_estimator.predictor is not None
    })


@app.route('/api/logs')
def get_logs():
    """Returns the live event alert stream."""
    return jsonify({
        "logs": event_logs[-30:],
        "total_alerts": alert_counter
    })


@app.route('/api/clear_logs', methods=['POST'])
def clear_logs():
    """Clears the event log history."""
    global event_logs, alert_counter
    event_logs.clear()
    alert_counter = 0
    return jsonify({"status": "cleared"})


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print("=" * 60)
    print(f" Real-Time Automated Exam Invigilation System running at:")
    print(f" http://127.0.0.1:{port}")
    print("=" * 60)
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
