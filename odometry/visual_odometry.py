import os
from typing import Sequence, ClassVar

import cv2
import numpy as np
import matplotlib.pyplot as plt


class VisualOdometry:
    draw_params: ClassVar = dict(matchColor=-1,  # draw matches in green color
                                 singlePointColor=None,
                                 matchesMask=None,  # draw only inliers
                                 flags=2)

    def __init__(self, camera_number: int, precision: float):
        self.K, self.P = self._load_calib(os.path.join("", 'calib.txt'))
        self.previous_image = None
        self.current_image = None

        self.capture = cv2.VideoCapture(camera_number)

        # Создание объекта ORB
        self.orb = cv2.ORB_create(3000) # noqa

        FLANN_INDEX_LSH = 6  # Указывает, что используем алгоритм LSH (6 ссответствует численному номеру алгоритма)
        # Параметры индексации
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        # Параметры поиска
        search_params = dict(checks=50)
        # Создание flann-матчера
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

        self.good_matches = []
        if not (0 < precision < 1):
            raise ValueError("Точность должна быть целым числом меньше единицы")
        self.precision = precision

    @staticmethod
    def _load_calib(filepath):
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

    @staticmethod
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix
        t (list): The translation vector

        Returns
        -------
        T (ndarray): The transformation matrix
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

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
        if gray:
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)

    def get_matches(self):
        # Find the keypoints and descriptors with ORB
        keypoints_from_previous_image, des1 = self.orb.detectAndCompute(self.previous_image, None)
        keypoints_from_current_image, des2 = self.orb.detectAndCompute(self.current_image, None)
        # Find matches
        matches = self.flann.knnMatch(des1, des2, k=2)

        if len(keypoints_from_previous_image) == 0 or len(keypoints_from_current_image) == 0:
            return None, None

        good = []
        try:
            for m, n in matches:
                if m.distance < 0.9 * n.distance:
                    good.append(m)
        except ValueError:
            pass

        img3 = cv2.drawMatches(  # noqa
            self.current_image,
            keypoints_from_previous_image,
            self.previous_image,
            keypoints_from_current_image,
            good,
            None,
            **self.draw_params
        )
        # cv2.imshow("image", img3)
        # cv2.waitKey(200)

        # Get the image points form the good matches
        matches_previous_image = np.float32([keypoints_from_previous_image[m.queryIdx].pt for m in good])
        matches_current_image = np.float32([keypoints_from_current_image[m.trainIdx].pt for m in good])

        return matches_previous_image, matches_current_image

    def decompose_essential_matrix(self, E, matches_previous_image, matches_current_image):
        """
        Decompose the Essential matrix
        """
        def sum_z_cal_relative_scale(R, t):
            # Get the transformation matrix
            T = self._form_transf(R, t)
            # Make the projection matrix
            P = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)

            # Triangulate the 3D points
            hom_Q1 = cv2.triangulatePoints(self.P, P, matches_previous_image.T, matches_current_image.T)
            # Also seen from cam 2
            hom_Q2 = np.matmul(T, hom_Q1)

            # Un-homogenize
            uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            # Find the number of points there has positive z coordinate in both cameras
            sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
            sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)

            # Form point pairs and calculate the relative scale
            relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)/
                                     np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1))
            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale

        # Decompose the essential matrix
        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)

        # Make a list of the different possible pairs
        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

        # Check which solution there is the right one
        z_sums = []
        relative_scales = []
        for R, t in pairs:
            z_sum, scale = sum_z_cal_relative_scale(R, t)
            z_sums.append(z_sum)
            relative_scales.append(scale)

        # Select the pair there has the most points with positive z coordinate
        right_pair_idx = np.argmax(z_sums)
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]
        R1, t = right_pair
        t = t * relative_scale

        return [R1, t]

    def get_pose(self, matches_previous_image, matches_current_image):
        """
        Calculate the transformation matrix
        """
        E, _ = cv2.findEssentialMat(matches_previous_image, matches_current_image, self.K, threshold=1)
        R, t = self.decompose_essential_matrix(E, matches_previous_image, matches_current_image)

        # Get transformation matrix
        transformation_matrix = self._form_transf(R, np.squeeze(t))
        return transformation_matrix


