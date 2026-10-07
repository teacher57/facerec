"""One way to read frames from either a USB webcam or the Raspberry Pi Camera Module."""
import cv2


class CameraError(Exception):
    pass


class UsbCamera:
    def __init__(self, index=0):
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise CameraError("Could not open the USB/built-in camera. On a Mac, check System "
                              "Settings > Privacy & Security > Camera and allow your terminal app.")

    def read(self):
        """Return the next BGR frame, or None if the camera stopped delivering frames."""
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self):
        self._cap.release()


class PiCamera:
    def __init__(self, picamera2_class):
        self._cam = picamera2_class()
        # Picamera2's "RGB888" format is stored as B, G, R per pixel: exactly what OpenCV expects.
        config = self._cam.create_video_configuration(main={"format": "RGB888", "size": (1280, 720)})
        self._cam.configure(config)
        self._cam.start()

    def read(self):
        return self._cam.capture_array("main")

    def close(self):
        self._cam.stop()
        self._cam.close()


def open_camera(kind="auto"):
    """Open a camera. kind: "picam", "usb", or "auto" (Pi camera if available, else USB)."""
    if kind in ("auto", "picam"):
        try:
            from picamera2 import Picamera2
            return PiCamera(Picamera2)
        except Exception as error:  # picamera2 missing, or no Pi camera connected
            if kind == "picam":
                raise CameraError(f"Could not open the Pi Camera Module: {error}") from error
    return UsbCamera()
