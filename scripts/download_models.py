"""
Model Weights Downloader & Setup Utility
Downloads or symlinks pretrained YOLOv3 weights and Dlib landmark predictors.
Author: Venkatrao Yasarapu
"""

import os
import sys
import shutil
import urllib.request

MODELS = {
    "shape_predictor_68_face_landmarks.dat": {
        "destination": "shape_predictor_68_face_landmarks.dat",
        "url": "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2",
        "description": "Dlib 68-Point Facial Landmark Shape Predictor"
    },
    "yolov3.weights": {
        "destination": os.path.join("models", "yolo-coco", "yolov3.weights"),
        "url": "https://pjreddie.com/media/files/yolov3.weights",
        "description": "YOLOv3 Pretrained Deep Learning Weights (248 MB)"
    }
}

LOCAL_SEARCH_PATHS = [
    r"C:\Users\vekat\Documents\Downloads\BATCH_18_CSE_Real Time Automated Exam Invigilation Using Computer Vision Techniques (1)\BATCH_18_CSE_Real Time Automated Exam Invigilation Using Computer Vision Techniques\project",
    r"C:\Users\vekat\.gemini\antigravity\scratch\temp_inspect\BATCH_18_CSE_Real Time Automated Exam Invigilation Using Computer Vision Techniques\project"
]


def progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, downloaded * 100.0 / total_size)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  Downloading: {percent:.1f}% ({mb_down:.1f} MB / {mb_total:.1f} MB)")
        sys.stdout.flush()


def find_local_copy(filename):
    for base in LOCAL_SEARCH_PATHS:
        for root, _, files in os.walk(base):
            if filename in files:
                return os.path.join(root, filename)
    return None


def main():
    print("=" * 65)
    print(" AI Exam Invigilator - Automated Model Setup")
    print("=" * 65)

    for filename, info in MODELS.items():
        dest = info["destination"]
        os.makedirs(os.path.dirname(dest) if os.path.dirname(dest) else ".", exist_ok=True)

        if os.path.exists(dest) and os.path.getsize(dest) > 1000000:
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f"[OK] {info['description']} is already present ({size_mb:.1f} MB).")
            continue

        print(f"\n[*] Setting up: {info['description']}")
        local_match = find_local_copy(filename)
        if local_match and os.path.exists(local_match):
            print(f"  Found local file at: {local_match}")
            print(f"  Copying to {dest}...")
            shutil.copy2(local_match, dest)
            print(f"[OK] Successfully copied from local storage!")
            continue

        print(f"  Downloading from official repository: {info['url']}")
        try:
            urllib.request.urlretrieve(info["url"], dest, progress_hook)
            print("\n[OK] Download completed successfully.")
        except Exception as err:
            print(f"\n[!] Download failed: {err}")
            print(f"    You can manually place {filename} into {dest}")

    print("\n" + "=" * 65)
    print(" Model setup process finished. You are ready to run app.py!")
    print("=" * 65)


if __name__ == "__main__":
    main()
