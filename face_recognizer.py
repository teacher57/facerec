"""Shared face-remembering code (OpenCV SFace): face -> 128 numbers, stored per person."""
from pathlib import Path

import cv2
import numpy as np

MODEL_PATH = Path(__file__).parent / "models" / "face_recognition_sface_2021dec.onnx"
KNOWN_FACES_PATH = Path(__file__).parent / "known_faces.npz"

# OpenCV's recommended cosine-similarity cutoff for SFace: at or above = same person.
MATCH_THRESHOLD = 0.363

_recognizer = None


def _get_recognizer():
    """Load SFace on first use, so commands like list_faces don't need the model."""
    global _recognizer
    if _recognizer is None:
        _recognizer = cv2.FaceRecognizerSF.create(str(MODEL_PATH), "")
    return _recognizer


def get_embedding(image, face):
    """Turn one detected face (a YuNet row) into 128 numbers, scaled to length 1."""
    recognizer = _get_recognizer()
    # alignCrop uses the 5 landmarks to rotate/crop the face the way SFace expects.
    aligned = recognizer.alignCrop(image, face)
    embedding = recognizer.feature(aligned).flatten()
    return embedding / np.linalg.norm(embedding)


def average_embeddings(embeddings):
    """Average several length-1 embeddings into one, scaled back to length 1."""
    mean = np.mean(embeddings, axis=0)
    return mean / np.linalg.norm(mean)


def load_known_faces():
    """Return {name: embedding} from known_faces.npz, or {} if nobody is enrolled yet."""
    if not KNOWN_FACES_PATH.exists():
        return {}
    data = np.load(KNOWN_FACES_PATH)
    return dict(zip(data["names"].tolist(), data["embeddings"]))


def save_person(name, embedding):
    """Add or replace one person in known_faces.npz."""
    known = load_known_faces()
    known[name] = embedding
    np.savez(KNOWN_FACES_PATH, names=np.array(list(known.keys())),
             embeddings=np.stack(list(known.values())))


def identify(image, face, known):
    """Return (name, probability) for one detected face.

    Known people get their similarity score; anyone else is ("stranger", face score).
    """
    name, similarity = best_match(get_embedding(image, face), known)
    if name is None:
        return "stranger", float(face[14])
    return name, similarity


def best_match(embedding, known):
    """Return (name, similarity) of the closest known person, or (None, best_similarity)."""
    best_name, best_score = None, -1.0
    for name, known_embedding in known.items():
        # Both are length 1, so the dot product is the cosine similarity (-1..1).
        score = float(np.dot(embedding, known_embedding))
        if score > best_score:
            best_name, best_score = name, score
    if best_score < MATCH_THRESHOLD:
        return None, best_score
    return best_name, best_score
