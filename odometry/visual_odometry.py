import os
from typing import Sequence, ClassVar

import cv2
import numpy as np


class VisualOdometry:

    draw_params: ClassVar = dict(matchColor=-1,  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=None,  # draw only inliers
                           flags=2)

    def __init__(self, camera_number: int):
        self.previous_image = None
        self.current_image = None

        self.capture = cv2.VideoCapture(camera_number)

        # Создание объекта ORB
        self.orb = cv2.ORB_create(3000)

        FLANN_INDEX_LSH = 6  # Указывает, что используем алгоритм LSH (6 ссответствует численному номеру алгоритма)
        # Параметры индексации
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        # Параметры поиска
        search_params = dict(checks=50)
        # Создание flann-матчера
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

        self.good_matches = []

    def filter_matches(self, matches: Sequence[Sequence[cv2.DMatch]]):
        try:
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    self.good_matches.append(m)
        except ValueError:
            pass


    def read_image(self, gray=False):
        if self.previous_image is None and self.current_image is None:
            _, self.current_image = self.capture.read()
        else:
            self.previous_image = self.current_image
            _, self.current_image = self.capture.read()

    def get_matches(self):
        # Find the keypoints and descriptors with ORB
        kp1, des1 = self.orb.detectAndCompute(self.previous_image, None)
        kp2, des2 = self.orb.detectAndCompute(self.current_image, None)
        # Find matches
        matches = self.flann.knnMatch(des1, des2, k=2)

        if len(kp1) == 0 or len(kp2) == 0:
            return None

        good = []
        try:
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    good.append(m)
        except ValueError:
            pass

        img3 = cv2.drawMatches(self.current_image, kp1, self.previous_image, kp2, good, None, **self.draw_params)
        cv2.imshow("image", img3)
        cv2.waitKey(200)

        # Get the image points form the good matches
        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])
        return q1, q2


od = VisualOdometry(0)

i = 0
while True:
    od.read_image(gray=True)


    od.get_matches()

    print(f"{i}-iteration \n\n\n")
    print(f"{od.good_matches}")
    i += 1

    # cv2.imshow('Video', od.current_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


