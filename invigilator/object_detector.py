"""
Prohibited Object Detection Module
Detects unauthorized cheating devices (smartphones, laptops, secondary monitors, books, notes) via YOLO DNN.
Author: Venkatrao Yasarapu
"""

import os
import cv2
import numpy as np


class ObjectDetector:
    """YOLO-based object detector to identify prohibited items in the exam environment."""

    # Prohibited COCO items during exams
    PROHIBITED_CLASSES = {
        "cell phone": "Smartphone / Mobile Device Detected",
        "laptop": "Unauthorized Secondary Laptop Detected",
        "book": "Exam Reference Book / Notes Detected",
        "remote": "Remote Control Device Detected"
    }

    def __init__(self,
                 config_path="models/yolo-coco/yolov3.cfg",
                 weights_path="models/yolo-coco/yolov3.weights",
                 labels_path="models/yolo-coco/coco.names",
                 confidence_threshold=0.5,
                 nms_threshold=0.4):
        self.config_path = config_path
        self.weights_path = weights_path
        self.labels_path = labels_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        self.net = None
        self.output_layers = []
        self.labels = []
        self.colors = []
        self.ready = False

        self._load_network()

    def _load_network(self):
        """Attempts to load Darknet YOLOv3 model."""
        if not os.path.exists(self.labels_path):
            print(f"[ObjectDetector] Labels file missing at {self.labels_path}")
            return

        with open(self.labels_path, "r", encoding="utf-8") as f:
            self.labels = [c.strip() for c in f.readlines()]

        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype="uint8")

        if not os.path.exists(self.config_path) or not os.path.exists(self.weights_path):
            print(f"[ObjectDetector] YOLO weights not found at {self.weights_path}.")
            print("[ObjectDetector] Run 'python scripts/download_models.py' to fetch pretrained weights.")
            return

        try:
            print("[ObjectDetector] Loading YOLOv3 deep neural network into memory...")
            self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)
            # Prefer OpenCV DNN backend
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            self.ready = True
            print("[ObjectDetector] YOLO model loaded successfully.")
        except Exception as exc:
            print(f"[ObjectDetector] Failed to initialize YOLO network: {exc}")

    def detect_objects(self, frame):
        """
        Runs object detection on the input frame.
        Returns:
            annotated_frame: Frame with bounding boxes drawn around detected items
            detected_items: List of dicts with detected item info
            violations: List of security alert strings
        """
        violations = []
        detected_items = []

        if not self.ready:
            return frame, detected_items, violations

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_outputs = self.net.forward(self.output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.confidence_threshold:
                    box = detection[0:4] * np.array([w, h, w, h])
                    center_x, center_y, box_w, box_h = box.astype("int")
                    x = int(center_x - (box_w / 2))
                    y = int(center_y - (box_h / 2))

                    boxes.append([x, y, int(box_w), int(box_h)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)

        person_count = 0
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, bw, bh = boxes[i]
                label_name = self.labels[class_ids[i]] if class_ids[i] < len(self.labels) else "unknown"
                conf = confidences[i]

                if label_name == "person":
                    person_count += 1

                detected_items.append({
                    "label": label_name,
                    "confidence": round(conf, 2),
                    "box": [x, y, bw, bh]
                })

                # Check if item is prohibited
                if label_name in self.PROHIBITED_CLASSES:
                    alert_msg = f"PROHIBITED ITEM: {self.PROHIBITED_CLASSES[label_name]} ({int(conf * 100)}% conf)"
                    violations.append(alert_msg)
                    color = (0, 0, 255)  # Red alert
                else:
                    color = [int(c) for c in self.colors[class_ids[i]]]

                # Draw bounding box and label badge
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
                tag = f"{label_name.upper()} {int(conf * 100)}%"
                cv2.putText(frame, tag, (x, max(20, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if person_count > 1:
            violations.append(f"Multiple individuals detected in exam space ({person_count} persons)")

        return frame, detected_items, violations
