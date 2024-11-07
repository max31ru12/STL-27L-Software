import os
from typing import ClassVar

import cv2
import numpy as np


class VisualOdometry:
    DRAW_PARAMS: ClassVar = dict(matchColor=-1,  # draw matches in green color
                                 singlePointColor=None,
                                 matchesMask=None,  # draw only inliers
                                 flags=2)

    FLANN_INDEX_LSH: ClassVar = 6
    INDEX_PARAMS = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    SEARCH_PARAMS = dict(checks=50)

    def __init__(self, camera_number: int, precision: float):

        # Калибровка камеры
        self.K, self.P = self._load_calib(os.path.join("", '../odometry/calib.txt'))

        # CURRENT STATE
        self.capture = cv2.VideoCapture(camera_number)
        self.current_image = None

        # PREVIOUS STATE
        self.previous_image = None
        self.previous_E = None

        # Создание объекта ORB
        self.orb = cv2.ORB_create(500)  # noqa
        # Создание flann-матчера
        self.flann = cv2.FlannBasedMatcher(indexParams=self.INDEX_PARAMS, searchParams=self.SEARCH_PARAMS)

        self.good_matches = []
        if not (0 < precision < 1):
            raise ValueError("Точность должна быть целым числом меньше единицы")
        self.precision = precision

    @staticmethod
    def _load_calib(filepath):
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]
        return K, P

    @staticmethod
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    @staticmethod
    def validate_descriptors(previous, current) -> bool:
        return (previous is None or current is None) or (len(previous) < 2 or len(current) < 2)

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

    # TESTED
    def read_image(self, gray=False, show=False):
        if self.previous_image is None and self.current_image is None:
            _, self.current_image = self.capture.read()
        else:
            self.previous_image = self.current_image
            _, self.current_image = self.capture.read()
        if gray:
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)

        if show:
            cv2.imshow("images", self.current_image)

    # TESTED
    def get_matches(self, show=False):
        # Find the keypoints and descriptors with ORB
        keypoints_from_previous_image, previous_descriptors = self.orb.detectAndCompute(self.previous_image, None)
        keypoints_from_current_image, current_descriptors = self.orb.detectAndCompute(self.current_image, None)
        # Find matches

        if self.validate_descriptors(previous_descriptors, current_descriptors):
            return None, None

        if len(keypoints_from_previous_image) == 0 or len(keypoints_from_current_image) == 0:
            return None, None

        matches = self.flann.knnMatch(previous_descriptors, current_descriptors, k=2)

        good = []

        try:
            for m, n in matches:
                if m.distance < 0.9 * n.distance:
                    good.append(m)
        except ValueError:
            pass

        if show:
            img3 = cv2.drawMatches(  # noqa
                self.current_image,
                keypoints_from_previous_image,
                self.previous_image,
                keypoints_from_current_image,
                good,
                None,
                **self.DRAW_PARAMS
            )
            cv2.imshow("image", img3)
            cv2.waitKey(200)

        # Get the image points form the good matches
        matches_previous_image = np.float32([keypoints_from_previous_image[m.queryIdx].pt for m in good])
        matches_current_image = np.float32([keypoints_from_current_image[m.trainIdx].pt for m in good])

        if len(matches_previous_image) < 5 or len(matches_current_image) < 5:
            return None, None

        return matches_previous_image, matches_current_image

    # TESTED
    def get_pose(self, matches_previous_image, matches_current_image):

        E, _ = cv2.findEssentialMat(matches_previous_image, matches_current_image, self.K, threshold=1)

        if E is None:
            print("Essential Matrix computation failed, using previous E.")
            E = self.previous_E
        else:
            self.previous_E = E  # Сохраняем текущее значение E

        R, t = self.decompose_essential_matrix(E, matches_previous_image, matches_current_image)

        # Get transformation matrix
        transformation_matrix = self._form_transf(R, np.squeeze(t))
        return transformation_matrix




