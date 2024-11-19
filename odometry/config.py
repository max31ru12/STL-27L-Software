import cv2
import numpy as np

# Калибровочные параметры камеры: (K - матрица калибровкиб P - )
# Формула проецирования точки x2d = Px3d
# Формула матрицы проекции P = K @ [R | t]

# Точки у чувака с ютуба: (Проекционная матрица)
# 707.0912  0.0  601.8873  0.0
# 0.0  707.0912  183.1104  0.0
# 0.0  0.0  1.0  0.0

# [ fx, 0., cx],
# [0., fy, cy],
# [0., 0., 1.]]

# fx и fy - фокусные расстояния по осям x и y (В ПИКСЕЛЯХ)
# сx и сy - координаты главной точки (В ПИКСЕЛЯХ)

FPS = int(24 / 6)
KEY_POINTS_QUANTITY = 500 * 2
ORB_DETECTOR = cv2.ORB_create(nfeatures=KEY_POINTS_QUANTITY)  # noqa

DISTANCE_BETWEEN_CAMERAS = 0.12  # м (в метрах)

FOCUS_DISTANCE = 3.67  # в мм (миллиметр),
EXTENSION = 3  # в МегаПикселях
PIXEL_REAL_DIMENSIONS = 1.4  # в мкм (микрометр) УТОЧНИТЬ

# в пикселях
fx, fy, cx, cy = 653.26, 653.79, 304.81, 229.71
K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0,  1]
])

# Левая камера - базовая, поэтому ее P - это [I | 0]
P_LEFT = K @ np.hstack((np.eye(3), np.zeros((3, 1))))  # в чем задается
P_RIGHT = K @ np.hstack((np.eye(3), [[-DISTANCE_BETWEEN_CAMERAS], [0], [0]]))

BF_MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
