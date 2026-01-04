# utils.py
import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("models/yolov8n.pt")

# COCO classes we care about
CHEATING_CLASSES = ["person", "cell phone"]

def detect_cheating(frame):
    results = model(frame, stream=True)

    person_count = 0
    phone_detected = False

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "person":
                person_count += 1
            if label == "cell phone":
                phone_detected = True

            # Draw bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cheating_flags = []

    if person_count > 1:
        cheating_flags.append("Multiple Persons Detected")

    if phone_detected:
        cheating_flags.append("Mobile Phone Detected")

    return frame, cheating_flags
