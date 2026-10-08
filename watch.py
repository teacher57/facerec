"""The camera loop shared by `run` (quiet) and `test` (-v window, -j printed job calls)."""
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2

from face_detector import GREEN, RED, draw_boxes, find_faces
from face_recognizer import identify, load_known_faces

LOGS_DIR = Path(__file__).parent / "logs"


def draw_faces(frame, faces, results):
    """Box every face: known people green with "name probability" below the box,
    strangers red with only their face score above it."""
    colors = [RED if name == "stranger" else GREEN for name, _ in results]
    draw_boxes(frame, faces, colors)
    frame_height = frame.shape[0]
    for face, (name, probability) in zip(faces, results):
        if name == "stranger":
            continue
        x, y, w, h = face[:4].astype(int)
        text_y = min(y + h + 22, frame_height - 5)
        cv2.putText(frame, f"{name} {probability:.2f}", (x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)


def _run_job(job_function, faces):
    """Run the user's job, printing its errors instead of stopping the camera loop."""
    try:
        job_function(faces)
    except Exception:
        print("job() raised an error:")
        traceback.print_exc()


def new_log_path():
    LOGS_DIR.mkdir(exist_ok=True)
    return LOGS_DIR / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.log"


def watch(camera, job_module, rate=5, display=False, log=False, run_job=True, print_jobs=False):
    """Check for faces `rate` times per second until Ctrl+C.

    Every check: identify each face, update its seen_for streak, and log it (if log=True:
    written to a new file in logs/ and printed in the terminal).
    If run_job, job_module.job(faces) runs in the background, at most once per SLEEP_TIME_SEC,
    and is skipped while a previous job call is still running. If print_jobs, every call is
    printed with a timestamp and the faces it received.
    """
    known = load_known_faces()
    print(f"Known people: {', '.join(known) if known else 'none yet (run enroll_face)'}")
    sleep_time_sec = getattr(job_module, "SLEEP_TIME_SEC", 2)

    log_file = None
    if log:
        log_path = new_log_path()
        log_file = open(log_path, "w")
        print(f"Logging to {log_path}")

    faces, results = [], []
    streaks = {}  # name -> number of checks in a row that name was seen
    last_check = 0.0
    last_job = float("-inf")
    job_thread = None
    print(f"Checking {rate} times per second. Press Ctrl+C to stop.")
    try:
        while True:
            frame = camera.read()
            if frame is None:
                print("Lost the camera feed (or camera permission was denied).")
                break

            now = time.monotonic()
            if now - last_check >= 1 / rate:
                last_check = now
                faces = find_faces(frame)
                results = [identify(frame, face, known) for face in faces]

                # A name not seen in this check loses its streak; all strangers share one.
                seen_names = {name for name, _ in results}
                streaks = {name: streaks.get(name, 0) + 1 for name in seen_names}
                job_faces = [{"name": name, "probability": probability, "seen_for": streaks[name]}
                             for name, probability in results]

                if log_file:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    for name, probability in results:
                        line = f"{name} {probability:.2f} {timestamp}"
                        log_file.write(line + "\n")
                        print(line)
                    log_file.flush()

                job_busy = job_thread is not None and job_thread.is_alive()
                if run_job and job_faces and not job_busy and now - last_job >= sleep_time_sec:
                    if print_jobs:
                        call_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        print(f"[{call_time}] job({job_faces})")
                    job_thread = threading.Thread(target=_run_job, args=(job_module.job, job_faces),
                                                  daemon=True)
                    job_thread.start()
                    last_job = now

            if display:
                draw_faces(frame, faces, results)
                cv2.imshow("Faces", frame)
                cv2.waitKey(1)  # lets the window refresh
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        camera.close()
        if display:
            cv2.destroyAllWindows()
        if log_file:
            log_file.close()
