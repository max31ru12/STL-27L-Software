from typing import Iterable

import cv2
import numpy as np


class ProcessingUtils:

    @classmethod
    def read_images(cls, left_camera, right_camera, gray=False):
        ret_left, left_frame = left_camera.read()
        ret_right, right_frame = right_camera.read()
        if gray:
            left_frame = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
            right_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        return ret_left, left_frame, ret_right, right_frame

    @classmethod
    def undistort_images(
            cls,
            left_frame: cv2.Mat,
            right_frame: cv2.Mat,
            camera_matrix: np.array,
            optimized_camera_matrix: np.array,
            distortion_coefficients: np.array,
            roi: Iterable,
    ):
        left_frame = cv2.undistort(left_frame, camera_matrix, distortion_coefficients, None, optimized_camera_matrix)
        right_frame = cv2.undistort(right_frame, camera_matrix, distortion_coefficients, None, optimized_camera_matrix)
        x, y, w, h = roi
        left_frame = left_frame[y:y + h, x:x + w]
        right_frame = right_frame[y:y + h, x:x + w]
        return left_frame, right_frame

    @classmethod
    def read_undistorted_images(
            cls,
            left_camera,
            right_camera,
            camera_matrix: np.array,
            optimized_camera_matrix: np.array,
            distortion_coefficients: np.array,
            roi: Iterable,
            gray=True
    ):
        ret_left, left_frame, ret_right, right_frame = cls.read_images(left_camera, right_camera, gray=gray)
        left_frame, right_frame = cls.undistort_images(
            left_frame, right_frame, camera_matrix, optimized_camera_matrix, distortion_coefficients, roi
        )

        return ret_left, left_frame, ret_right, right_frame

    @classmethod
    def disparity_to_3d(cls, disparity, focal_length, baseline, cx, cy):
        """
        # Преобразование карты диспаритета в 3D-точки
        """
        h, w = disparity.shape
        Q = np.zeros((h, w, 3), dtype=np.float32)
        Z = (focal_length * baseline) / (disparity + 1e-6)  # Добавляем epsilon, чтобы избежать деления на 0
        X = (np.tile(np.arange(w), (h, 1)) - cx) * Z / focal_length
        Y = (np.tile(np.arange(h), (w, 1)).T - cy) * Z / focal_length
        Q[:, :, 0] = X
        Q[:, :, 1] = Y
        Q[:, :, 2] = Z
        return Q

    @classmethod
    def detect_keypoints_with_tiling(cls, image, tile_height, tile_width, max_features_per_tile=10):
        height, width = image.shape
        fast_detector = cv2.FastFeatureDetector_create()
        keypoints = []

        for y in range(0, height, tile_height):
            for x in range(0, width, tile_width):
                patch = image[y:y + tile_height, x:x + tile_width]
                tile_keypoints = fast_detector.detect(patch)

                for kp in tile_keypoints:
                    kp.pt = (kp.pt[0] + x, kp.pt[1] + y)

                if len(tile_keypoints) > max_features_per_tile:
                    tile_keypoints = sorted(tile_keypoints, key=lambda k: -k.response)[:max_features_per_tile]

                keypoints.extend(tile_keypoints)

        return keypoints

    @classmethod
    def track_features_lk(cls, previous_frame, current_frame, keypoints, lk_params):
        keypoints_np = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(previous_frame, current_frame, keypoints_np, None, **lk_params)

        valid_keypoints_prev = keypoints_np[status == 1]
        valid_keypoints_next = new_points[status == 1]
        return valid_keypoints_prev, valid_keypoints_next


class Filters:

    @classmethod
    def filter_points(cls, good_prev, good_curr):
        height, width = 480, 640
        valid_indices = (
                (good_prev[:, 1] >= 0) & (good_prev[:, 1] < height) &
                (good_prev[:, 0] >= 0) & (good_prev[:, 0] < width)
        )
        return good_prev[valid_indices], good_curr[valid_indices]
