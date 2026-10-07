"""Remember a new person, from the camera or from a folder of pictures."""
import time
from pathlib import Path

import cv2

from face_detector import draw_boxes, find_faces
from face_recognizer import average_embeddings, get_embedding, save_person

SHOTS_NEEDED = 15
SECONDS_BETWEEN_SHOTS = 0.5
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def draw_status(frame, message, shots_taken):
    """Draw a progress bar and a status message along the bottom of the frame."""
    height, width = frame.shape[:2]
    bar_left, bar_right = 20, width - 20
    bar_top, bar_bottom = height - 40, height - 20
    filled = bar_left + int((bar_right - bar_left) * shots_taken / SHOTS_NEEDED)
    cv2.rectangle(frame, (bar_left, bar_top), (bar_right, bar_bottom), (255, 255, 255), 2)
    cv2.rectangle(frame, (bar_left, bar_top), (filled, bar_bottom), (0, 255, 0), -1)
    cv2.putText(frame, f"{message}  ({shots_taken}/{SHOTS_NEEDED})", (bar_left, bar_top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def enroll_from_camera(name, camera):
    """Take 15 shots, one every 0.5 s, only while exactly one face is visible; then save.

    Returns True if the person was saved, False if cancelled (Ctrl+C) or the camera failed.
    """
    embeddings = []
    last_shot = 0.0
    print(f"Remembering {name}. Look at the camera and move your head a little. Ctrl+C cancels.")
    try:
        while len(embeddings) < SHOTS_NEEDED:
            frame = camera.read()
            if frame is None:
                print("Lost the camera feed (or camera permission was denied).")
                return False

            faces = find_faces(frame)
            if len(faces) == 0:
                message = "No face found - waiting"
            elif len(faces) > 1:
                message = f"Paused: leave only {name} in the frame"
            else:
                message = f"Remembering {name} - move your head a little"
                now = time.monotonic()
                if now - last_shot >= SECONDS_BETWEEN_SHOTS:
                    # Take the embedding from the clean frame, before anything is drawn on it.
                    embeddings.append(get_embedding(frame, faces[0]))
                    last_shot = now

            draw_boxes(frame, faces)
            draw_status(frame, message, len(embeddings))
            cv2.imshow("Enroll face", frame)
            cv2.waitKey(1)
    except KeyboardInterrupt:
        print("\nCancelled. Nothing was saved.")
        return False
    finally:
        camera.close()
        cv2.destroyAllWindows()

    save_person(name, average_embeddings(embeddings))
    print(f"Saved {name} ({SHOTS_NEEDED} shots averaged into one face).")
    return True


def enroll_from_folder(name, folder):
    """Average every picture in the folder that shows exactly one face; then save.

    Returns True if at least one usable picture was found and the person was saved.
    """
    if not Path(folder).is_dir():
        print(f"Not a folder: {folder}")
        return False
    paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    embeddings = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"  skipped {path.name}: could not read it")
            continue
        faces = find_faces(image)
        if len(faces) != 1:
            print(f"  skipped {path.name}: {len(faces)} faces (need exactly 1)")
            continue
        embeddings.append(get_embedding(image, faces[0]))
        print(f"  used    {path.name} (face score {faces[0][14]:.2f})")

    if not embeddings:
        print(f"No usable pictures in {folder}. Nothing was saved.")
        return False
    save_person(name, average_embeddings(embeddings))
    print(f"Saved {name} ({len(embeddings)} of {len(paths)} pictures averaged into one face).")
    return True
