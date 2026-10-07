"""Your code: what to do when the camera sees faces. Edit freely, then restart `run`.

job(faces) is called in the background, at most once every SLEEP_TIME_SEC seconds,
whenever at least one face is seen. `faces` is a list with one dict per face, e.g.

    [{"name": "Arnold", "probability": 0.71, "seen_for": 12},
     {"name": "stranger", "probability": 0.93, "seen_for": 3}]

- probability: match score for known people, face-detection score for strangers
- seen_for: how many checks in a row this name has been seen (resets when it is missing);
  use it to ignore one-off mistakes, e.g. only act when seen_for >= 5
"""

SLEEP_TIME_SEC = 2


def job(faces):
    pass
    # Example:
    # for face in faces:
    #     if face["name"] == "Arnold" and face["seen_for"] >= 5:
    #         print("Arnold here!")
