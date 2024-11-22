# src/utils/camera.py
import cv2
import numpy as np
from .config import IMG_SIZE, FPS_VIDEO_READING


class Camera:
    def __init__(self, img_size: int = IMG_SIZE, fps: int = FPS_VIDEO_READING):
        self.img_size = img_size
        self.fps = fps
        self.cap = self._initialize_camera()

    def _initialize_camera(self) -> cv2.VideoCapture:
        """
        Init the camera object with the desired settings

        :return cap: Camera object
        """
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_size)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_size)
        return cap

    def read_frame(self) -> np.ndarray:
        """
        Read a frame from the camera and return it in RGB format

        :return frame: Frame video in RGB format
        """
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Error reading the frame")
        return frame[:, :, [2, 1, 0]]
