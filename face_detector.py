"""Shared face-finding code (OpenCV YuNet), used by every script in this project."""
from pathlib import Path

import cv2

MODEL_PATH = Path(__file__).parent / "models" / "face_detection_yunet_2023mar.onnx"

# Created once and reused; the input size is updated per image in find_faces().
_detector = cv2.FaceDetectorYN.create(str(MODEL_PATH), "", (320, 320))


def find_faces(image):
    """Return a (num_faces, 15) array of detected faces, or an empty list if none.

    Each row: [x, y, w, h, 5 landmark (x, y) pairs, confidence score].
    """
    height, width = image.shape[:2]
    _detector.setInputSize((width, height))
    _, faces = _detector.detect(image)
    return faces if faces is not None else []


def draw_boxes(image, faces):
    """Draw a green box around each face with its confidence score above it, in place."""
    for face in faces:
        x, y, w, h = face[:4].astype(int)
        score = face[14]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"{score:.2f}", (x, max(y - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
