import cv2
import numpy as np


# Калибровочные параметры камеры: (K - матрица калибровкиб P - )
# Формула проецирования точки x2d = Px3d
# Формула матрицы проекции P = K @ [R | t]

# fx и fy - фокусные расстояния по осям x и y (В ПИКСЕЛЯХ)
# сx и сy - координаты главной точки (В ПИКСЕЛЯХ)

# в пикселях до
# fx, fy, cx, cy = 653.26, 653.79, 304.81, 229.71 Старые значения с 4 изображений шахматки

fx, fy, cx, cy = 551.25345092, 551.43716708, 303.15262187, 220.81193167
# fx, fy, cx, cy = 1322.95, 1321.56, 960, 540 Это мне нагенерила нейросеть
CAMERA_MATRIX = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0,  1]
])


FPS = int(24 / 12)
KEY_POINTS_QUANTITY = 500 * 2
ORB_DETECTOR = cv2.ORB_create(nfeatures=KEY_POINTS_QUANTITY)  # noqa

DISTANCE_BETWEEN_CAMERAS = 0.12  # м (в метрах)

FOCUS_DISTANCE = 3.67   # в мм (миллиметр),
FOCUS_DISTANCE_METERS = FOCUS_DISTANCE / 100 / 100  # в метрах
EXTENSION = 3  # в МегаПикселях
PIXEL_REAL_DIMENSIONS = 1.4  # в мкм (микрометр) УТОЧНИТЬ

k1, k2, p1, p2, k3 = 0.02872523, -0.09701738, -0.00640777, -0.00613953, 0.01908023
DISTORTION_COEFFICIENTS = np.array([[k1, k2, p1, p2, k3]])

opt_fx, opt_fy, opt_cx, opt_cy = 539.88840599, 538.8285721, 297.68377963, 216.61230747
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
    "minDisparity": 0,
    "numDisparities": 16 * 5,  # Кратное 16
    "blockSize": 15,
    "uniquenessRatio": 10,
    "speckleWindowSize": 100,
    "speckleRange": 32,
    "disp12MaxDiff": 1,
    "preFilterCap": 63,
    "P1": 8 * 15 ** 2,
    "P2": 32 * 15 ** 2,
    "mode": cv2.STEREO_SGBM_MODE_SGBM_3WAY,
}

# All points matchers
BM_engine = cv2.StereoBM_create(numDisparities=16 * 5, blockSize=15)  # noqa
SGBM_engine = cv2.StereoSGBM_create(**SGBM_PARAMS)  # noqa

# Lucas Kanade parameters
LK_PARAMS = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

