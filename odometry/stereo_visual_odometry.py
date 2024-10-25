from os import PathLike
from typing import Any

import cv2
import numpy as np


Frame = cv2.Mat | np.ndarray[Any, np.dtype[np.generic]] | np.ndarray


class StereoVisualOdometry:  # noqa

    def __init__(self, left_camera_number: int = 1, right_camera_number: int = 2):
        self.K_left, self.P_left = self.__load_calib("calib.txt")
        self.K_right, self.P_right = self.__load_calib("calib.txt")

        print(type(self.K_left))
        print(type(self.P_left))

        # CURRENT STATE
        self.left_capture = cv2.VideoCapture(left_camera_number)
        self.right_capture = cv2.VideoCapture(right_camera_number)

        self.left_current_frame = None
        self.right_current_frame = None

        # PREVIOUS STATE
        self.left_previous_frame = None
        self.right_previous_frame = None

    @staticmethod
    def __load_calib(filepath: str | PathLike) -> tuple[np.ndarray, np.ndarray]:
        """
        Loads the calibration of the camera
        Parameters
        ----------
        filepath (str): The file path to the camera file

        Returns
        -------
        K (ndarray): Intrinsic parameters
        P (ndarray): Projection matrix
        """
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]
        return K, P

    @property
    def current_frames_is_none(self) -> bool:
        return self.left_current_frame is None and self.right_current_frame

    @property
    def previous_frames_is_none(self) -> bool:
        return self.left_previous_frame is None and self.right_previous_frame

    def read_images(self, gray=False, show=False):

        if self.current_frames_is_none and self.previous_frames_is_none:
            self.left_current_frame: Frame = self.left_capture.read()[1]
            self.right_current_frame: Frame = self.right_capture.read()[1]

        else:
            # make current state previous
            self.left_previous_frame = self.left_current_frame
            self.right_previous_frame = self.right_current_frame
            # get new state
            self.left_current_frame: Frame = self.left_capture.read()[1]
            self.right_current_frame: Frame = self.right_capture.read()[1]

        if gray:
            self.left_current_frame = cv2.cvtColor(self.left_current_frame, cv2.COLOR_BGR2GRAY)
            self.right_current_frame = cv2.cvtColor(self.right_current_frame, cv2.COLOR_BGR2GRAY)

        if show:
            cv2.imshow("Left?", self.left_current_frame)
            cv2.imshow("Right", self.right_current_frame)
            cv2.waitKey(200)


od = StereoVisualOdometry(1, 2)

while True:
    od.read_images(show=True, gray=True)
