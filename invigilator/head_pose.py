"""
Head Pose Estimation & Facial Landmark Analysis Module
Calculates Euler angles (Pitch, Yaw, Roll) using 68-point Dlib landmarks and cv2.solvePnP.
Author: Venkatrao Yasarapu
"""

import os
import math
import cv2
import numpy as np

try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False


class HeadPoseEstimator:
    """Estimates head pose (Pitch, Yaw, Roll) and tracks suspicious candidate head movements."""

    def __init__(self, landmark_model_path="shape_predictor_68_face_landmarks.dat"):
        self.landmark_model_path = landmark_model_path
        self.dlib_available = DLIB_AVAILABLE
        self.detector = None
        self.predictor = None
        self.haar_cascade = None

        # 3D Model Anthropometric Landmarks
        self.model_points_3d = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)

        # 68-point landmark indices corresponding to the 3D model points
        self.landmark_indices_2d = np.array([33, 8, 36, 45, 48, 54], dtype=np.uint32)

        self._initialize_detectors()

    def _initialize_detectors(self):
        """Initializes Dlib facial detector and predictor, or falls back to OpenCV Haar."""
        if self.dlib_available and os.path.exists(self.landmark_model_path):
            try:
                self.detector = dlib.get_frontal_face_detector()
                self.predictor = dlib.shape_predictor(self.landmark_model_path)
                print(f"[HeadPoseEstimator] Loaded Dlib model from {self.landmark_model_path}")
                return
            except Exception as exc:
                print(f"[HeadPoseEstimator] Dlib model load failed: {exc}. Falling back to OpenCV.")

        # Fallback Haar Cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.haar_cascade = cv2.CascadeClassifier(cascade_path)
        print("[HeadPoseEstimator] Initialized OpenCV Haar Cascade for face tracking.")

    def _get_camera_matrix(self, img_shape):
        """Builds an approximated camera intrinsic matrix based on image dimensions."""
        h, w = img_shape[:2]
        focal_length = w
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        return camera_matrix

    def _rotation_matrix_to_euler_angles(self, R):
        """Decomposes a 3x3 rotation matrix into Euler angles (Pitch, Yaw, Roll) in degrees."""
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            pitch = math.atan2(R[2, 1], R[2, 2])
            yaw = math.atan2(-R[2, 0], sy)
            roll = math.atan2(R[1, 0], R[0, 0])
        else:
            pitch = math.atan2(-R[1, 2], R[1, 1])
            yaw = math.atan2(-R[2, 0], sy)
            roll = 0.0

        return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)

    def process_frame(self, frame):
        """
        Analyzes the candidate's face in the frame.
        Returns:
            processed_frame: Annotated OpenCV frame
            pose_data: dict with pitch, yaw, roll, and detection status
            violations: list of string alerts
        """
        violations = []
        pose_data = {
            "face_detected": False,
            "face_count": 0,
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0,
            "status": "No Face Detected"
        }

        h, w = frame.shape[:2]

        if self.predictor is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray, 0)
            pose_data["face_count"] = len(faces)

            if len(faces) == 0:
                violations.append("Candidate face not visible in frame")
                cv2.putText(frame, "STATUS: Candidate Not Detected!", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return frame, pose_data, violations

            if len(faces) > 1:
                violations.append(f"Multiple persons detected in camera feed ({len(faces)} faces)")
                cv2.putText(frame, f"ALERT: Multiple Faces Detected ({len(faces)})", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            pose_data["face_detected"] = True
            primary_face = faces[0]
            fx1, fy1, fx2, fy2 = primary_face.left(), primary_face.top(), primary_face.right(), primary_face.bottom()
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)

            shape = self.predictor(gray, primary_face)
            landmarks_2d = np.zeros((68, 2), dtype=np.float64)
            for i in range(68):
                landmarks_2d[i] = (shape.part(i).x, shape.part(i).y)

            # Draw key facial landmark points
            for (lx, ly) in landmarks_2d:
                cv2.circle(frame, (int(lx), int(ly)), 1, (0, 255, 255), -1)

            # Perspective-n-Point pose calculation
            image_points = landmarks_2d[self.landmark_indices_2d]
            camera_matrix = self._get_camera_matrix(frame.shape)
            dist_coeffs = np.zeros((4, 1))

            success, rot_vec, trans_vec = cv2.solvePnP(
                self.model_points_3d,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:
                rot_mat, _ = cv2.Rodrigues(rot_vec)
                pitch, yaw, roll = self._rotation_matrix_to_euler_angles(rot_mat)
                pose_data["pitch"] = round(pitch, 1)
                pose_data["yaw"] = round(yaw, 1)
                pose_data["roll"] = round(roll, 1)

                # Project 3D nose direction vector to visualize head direction
                nose_end_point3D = np.array([(0.0, 0.0, 1000.0)])
                nose_end_point2D, _ = cv2.projectPoints(
                    nose_end_point3D, rot_vec, trans_vec, camera_matrix, dist_coeffs
                )
                p1 = (int(image_points[0][0]), int(image_points[0][1]))
                p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
                cv2.line(frame, p1, p2, (255, 0, 0), 2)

                # Suspicious gaze & pose classification
                if yaw > 28:
                    violations.append("Candidate looking right (Suspicious Head Movement)")
                elif yaw < -28:
                    violations.append("Candidate looking left (Suspicious Head Movement)")

                if pitch < -25:
                    violations.append("Candidate looking down (Possible Notes/Device)")
                elif pitch > 30:
                    violations.append("Candidate looking up away from screen")

                status_text = "FOCUSED" if not violations else "SUSPICIOUS POSE"
                pose_data["status"] = status_text
                hud_color = (0, 255, 0) if not violations else (0, 0, 255)

                # HUD Overlay
                cv2.putText(frame, f"Pitch: {pitch:.1f}  Yaw: {yaw:.1f}  Roll: {roll:.1f}",
                            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_color, 2)
                cv2.putText(frame, f"Pose Status: {status_text}",
                            (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_color, 2)

        else:
            # Fallback with OpenCV Haar Cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.haar_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
            pose_data["face_count"] = len(faces)

            if len(faces) == 0:
                violations.append("Candidate face not visible in frame")
                cv2.putText(frame, "STATUS: Face Not Detected!", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                pose_data["face_detected"] = True
                pose_data["status"] = "Face Detected (Haar Mode)"
                for (x, y, fw, fh) in faces:
                    cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
                if len(faces) > 1:
                    violations.append(f"Multiple persons detected ({len(faces)} faces)")

        return frame, pose_data, violations
