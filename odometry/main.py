import sys
import time

import numpy as np
import cv2
from loguru import logger

from odometry.config import ORB_DETECTOR, BF_MATCHER, P_LEFT, P_RIGHT, FPS
from odometry.odometry_utils import (
    DebugTools,
    keypoints_is_empty,
    triangulate_points,
    estimate_motion, get_matched_3D_points, read_images, filter_image,
)

logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")
LEFT_CAMERA = cv2.VideoCapture(0)
RIGHT_CAMERA = cv2.VideoCapture(2)

if not LEFT_CAMERA.isOpened() or not RIGHT_CAMERA.isOpened():
    logger.error("Не удалось открыть одну из камер")

counter = 0

# Инициализация предыдущего состояния

previous_left_frame, previous_right_frame = None, None
initial_position = np.array([0, 0, 0, 1])
current_transform = np.eye(4)

estimated_path = {
    "x": [0],
    "y": [0],
    "z": [0],
}

while True:

    time.sleep(1 / FPS)

    ret_left, left_frame, ret_right, right_frame = read_images(LEFT_CAMERA, RIGHT_CAMERA, gray=True)

    if not ret_left or not ret_right:
        logger.error("Не удалось захватить кадры с обеих камер")
        continue

    # Для обработки первого получения изображений
    if not counter:
        previous_left_frame, previous_right_frame = left_frame, right_frame
        counter += 1
        continue

    previous_keypoints_left, previous_descriptors_left = ORB_DETECTOR.detectAndCompute(previous_left_frame, None)
    previous_keypoints_right, previous_descriptors_right = ORB_DETECTOR.detectAndCompute(previous_right_frame, None)

    keypoints_left, descriptors_left = ORB_DETECTOR.detectAndCompute(left_frame, None)
    keypoints_right, descriptors_right = ORB_DETECTOR.detectAndCompute(right_frame, None)

    # DON'T UPDATE PREVIOUS STATE
    if keypoints_is_empty(keypoints_left, keypoints_right, descriptors_left, descriptors_right):
        logger.warning("Не найдены ключевые точки")
        continue

    previous_matches = BF_MATCHER.match(previous_descriptors_left, previous_descriptors_right)
    matches = BF_MATCHER.match(descriptors_left, descriptors_right)

    previous_sorted_matches = sorted(previous_matches, key=lambda x: x.distance)
    sorted_matches = sorted(matches, key=lambda x: x.distance)

    previous_left_and_current_left_matches = BF_MATCHER.match(previous_descriptors_left, descriptors_left)

    previous_points_3D = triangulate_points(
        P_LEFT, P_RIGHT, previous_keypoints_left, previous_keypoints_right, previous_sorted_matches, log=True
    )
    points_3D = triangulate_points(
        P_LEFT, P_RIGHT, keypoints_left, keypoints_right, sorted_matches, log=True
    )

    try:
        previous_points_3D, points_3D = get_matched_3D_points(
            previous_keypoints_3D=previous_points_3D,
            current_keypoints_3D=points_3D,
            previous_matches=previous_sorted_matches,
            matches=sorted_matches,
            previous_left_and_current_left_matches=previous_left_and_current_left_matches,
        )
    except Exception as e:
        continue

    if previous_points_3D.size > 0 and points_3D.size > 0:
        R, t = estimate_motion(previous_points_3D, points_3D)
        logger.info(f"Изменение позы:\nМатрица вращения:\n{R}\nВектор трансляции:\n{t}")


    previous_left_frame, previous_right_frame = left_frame, right_frame
    cv2.waitKey(1)
