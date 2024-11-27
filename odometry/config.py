import cv2
import numpy as np
from pydantic_settings import BaseSettings


# Калибровочные параметры камеры: (K - матрица калибровкиб P - )
# Формула проецирования точки x2d = Px3d
# Формула матрицы проекции P = K @ [R | t]

# fx и fy - фокусные расстояния по осям x и y (В ПИКСЕЛЯХ)
# сx и сy - координаты главной точки (В ПИКСЕЛЯХ)

# в пикселях до
# fx, fy, cx, cy = 653.26, 653.79, 304.81, 229.71 Старые значения с 4 изображений шахматки

# Предыдущие параметры
fx, fy, cx, cy = 551.25345092, 551.43716708, 303.15262187, 220.81193167
CAMERA_MATRIX = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0,  1]
])


FPS = int(24 / 6)
KEY_POINTS_QUANTITY = 500 * 2
ORB_DETECTOR = cv2.ORB_create(nfeatures=KEY_POINTS_QUANTITY)  # noqa

DISTANCE_BETWEEN_CAMERAS = 0.12  # м (в метрах)

FOCUS_DISTANCE = 3.67   # в мм (миллиметр),
FOCUS_DISTANCE_METERS = FOCUS_DISTANCE / 100 / 100  # в метрах
EXTENSION = 3  # в МегаПикселях
PIXEL_REAL_DIMENSIONS = 1.4  # в мкм (микрометр) УТОЧНИТЬ

k1, k2, p1, p2, k3 = 0.02872523, -0.09701738, -0.00640777, -0.00613953, 0.01908023
DISTORTION_COEFFICIENTS = np.array([[k1, k2, p1, p2, k3]])

fx, fy, cx, cy = 539.88840599, 538.8285721, 297.68377963, 216.61230747
OPTIMIZED_CAMERA_MATRIX = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0,  1]
])
# Область интереса с полезной инфомацией после фильтрации искажений
ROI = (4, 4, 626, 466)

# Левая камера - базовая, поэтому ее P - это [I | 0]
P_LEFT = CAMERA_MATRIX @ np.hstack((np.eye(3), np.zeros((3, 1))))  # в чем задается
P_RIGHT = CAMERA_MATRIX @ np.hstack((np.eye(3), [[-DISTANCE_BETWEEN_CAMERAS], [0], [0]]))

# Keypoints matcher
BF_MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

SGBM_PARAMS = {
    "minDisparity": 0,  # Минимальное значение диспарити
    "numDisparities": 16 * 5,  # Количество диспарити, кратное 16
    "blockSize": 7,  # Размер блока для сравнения
    "uniquenessRatio": 10,  # Процент уникальности для фильтрации шумов
    "speckleWindowSize": 100,  # Максимальный размер области шумов
    "speckleRange": 32,  # Максимальная разница диспарити для шумов
    "disp12MaxDiff": 1,  # Максимальная разница в диспарити между левым и правым изображениями
    "preFilterCap": 63,  # Порог фильтра для предварительной обработки
    "P1": 8 * 3 * 7 ** 2,  # Параметр P1 для SGBM 7 - block_size
    "P2": 32 * 3 * 7 ** 2,  # Параметр P2 для SGBM 7 - block_size
    "mode": cv2.STEREO_SGBM_MODE_SGBM_3WAY,  # Точный 3-проходной режим
}

# All points matchers
STEREO_MATCHER = cv2.StereoBM_create(numDisparities=16 * 5, blockSize=15)  # noqa
SEMI_GLOBAL_STEREO_MATCHER = cv2.StereoSGBM_create(**SGBM_PARAMS)  # noqa

