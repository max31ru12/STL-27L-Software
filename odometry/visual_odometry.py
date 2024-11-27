import sys
import time

import numpy as np
import cv2
from loguru import logger

from odometry.config import (
    ORB_DETECTOR,
    BF_MATCHER,
    P_LEFT,
    P_RIGHT,
    FPS,
    STEREO_MATCHER,
    SEMI_GLOBAL_STEREO_MATCHER, FOCUS_DISTANCE, DISTANCE_BETWEEN_CAMERAS, FOCUS_DISTANCE_METERS,
)
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

_, previous_left_frame, _, previous_right_frame = read_images(LEFT_CAMERA, RIGHT_CAMERA, gray=True)
initial_position = np.array([0, 0, 0, 1])
current_transform = np.eye(4)

estimated_path = {
    "x": [0],
    "y": [0],
    "z": [0],
}

while True:

    time.sleep(1 / FPS)

    ret_left, left_frame, ret_right, right_frame = read_images(LEFT_CAMERA, RIGHT_CAMERA, gray=True, filtered=True)

    if not ret_left or not ret_right:
        logger.error("Не удалось захватить кадры с обеих камер")
        continue

    # получение карты диспаритета: disparity = xL - xR
    # диспаритет - разность между X координатами соответствующих точек двух изображений
    disparity = SEMI_GLOBAL_STEREO_MATCHER.compute(left_frame, right_frame).astype(np.float32) / 16.0

    DebugTools.show_disparity_map(disparity, normalized=True)

    # Вычисление глубины для каждого пикселя
    depth_map = (FOCUS_DISTANCE_METERS * DISTANCE_BETWEEN_CAMERAS) / (disparity + 1e-6)

    # DebugTools.show_disparity_map(depth_map, normalized=True)


    # === 3. Выделение ключевых точек ===
    keypoints, descriptors = ORB_DETECTOR.detectAndCompute(left_frame, None)  # Только на левом кадре

    # Преобразование ключевых точек в NumPy массив
    points_2d_current = np.array([kp.pt for kp in keypoints], dtype=np.float32)

    # === 4. Оптический поток для отслеживания ===
    # points_2d_previous, status, _ = cv2.calcOpticalFlowPyrLK(previous_left_frame, left_frame, points_2d_current, None)

    # Фильтрация точек
    # valid_points_current = points_2d_current[status == 1]
    # valid_points_previous = points_2d_previous[status == 1]

    # DON'T UPDATE PREVIOUS STATE
    # if keypoints_is_empty(keypoints_left, keypoints_right, descriptors_left, descriptors_right):
    #     logger.warning("Не найдены ключевые точки")
    #     continue

    previous_left_frame, previous_right_frame = left_frame, right_frame

    cv2.waitKey(1)


    counter += 1