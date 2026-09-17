"""
Face Verification and Intruder Detection Module
Validates registered candidate identity and detects unauthorized substitute test-takers.
Author: Venkatrao Yasarapu
"""

import os
import cv2
import numpy as np

try:
    import face_recognition
    FACE_REC_AVAILABLE = True
except ImportError:
    FACE_REC_AVAILABLE = False


class FaceVerifier:
    """Verifies exam candidate identity against registered photos."""

    def __init__(self, known_faces_dir="known_faces"):
        self.known_faces_dir = known_faces_dir
        self.known_encodings = []
        self.known_names = []
        self.enabled = FACE_REC_AVAILABLE
        self.load_known_faces()

    def load_known_faces(self):
        """Loads and encodes all candidate photos from the known_faces directory."""
        if not self.enabled:
            print("[FaceVerifier] face_recognition library not available. Running in pass-through mode.")
            return

        if not os.path.isdir(self.known_faces_dir):
            os.makedirs(self.known_faces_dir, exist_ok=True)

        valid_extensions = {".jpg", ".jpeg", ".png"}
        for filename in os.listdir(self.known_faces_dir):
            name, ext = os.path.splitext(filename)
            if ext.lower() in valid_extensions:
                file_path = os.path.join(self.known_faces_dir, filename)
                try:
                    image = face_recognition.load_image_file(file_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        self.known_encodings.append(encodings[0])
                        self.known_names.append(name.capitalize())
                        print(f"[FaceVerifier] Registered candidate face: {name.capitalize()}")
                    else:
                        print(f"[FaceVerifier] No face found in {filename}, skipping.")
                except Exception as err:
                    print(f"[FaceVerifier] Error loading candidate photo {filename}: {err}")

    def verify_frame(self, frame):
        """
        Detects faces in frame and validates against known candidate encodings.
        Returns:
            annotated_frame: Frame with labeled candidate/intruder bounding boxes
            candidate_names: List of recognized names
            violations: List of security alert messages
        """
        violations = []
        candidate_names = []

        if not self.enabled or not self.known_encodings:
            return frame, candidate_names, violations

        # Downscale for faster real-time FPS
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(self.known_encodings, face_encoding, tolerance=0.55)
            name = "Unknown Intruder"
            box_color = (0, 0, 255)

            face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_names[best_match_index]
                    box_color = (0, 255, 0)

            candidate_names.append(name)

            # Scale back coordinates
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            if name == "Unknown Intruder":
                violations.append("Unauthorized substitute or unrecognized face detected in frame")

            # Draw labeled box
            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
            cv2.rectangle(frame, (left, bottom - 30), (right, bottom), box_color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 8),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

        return frame, candidate_names, violations
