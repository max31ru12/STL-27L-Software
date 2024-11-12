import sys

import numpy as np
import cv2

from loguru import logger

from odometry.odometry_utils import (
    keypoints_is_empty,
    print_keypoints,
    draw_with_keypoints,
    draw_keypoints_matches,
    triangulate_points,
    estimate_motion, get_matched_3D_points
)

logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")

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

CAMERA_CALIBRATION_PARAMETERS = {
    "K_LEFT": np.array([[653.2621743,   0.,             304.81329016],
                        [0.,            653.7921645,    229.71279112],
                        [0.,            0.,             1.]]),
    "K_RIGHT": np.array(np.array([[653.2621743,   0.,             304.81329016],
                        [0.,            653.7921645,    229.71279112],
                        [0.,            0.,             1.]]),),
    "FOCUS_DISTANCE": 3.67,  # в мм (миллиметр),
    "EXTENSION": 3,  # в МегаПикселях
    "PIXEL_REAL_DIMENSIONS": 1.4,  # в мкм (микрометр) УТОЧНИТЬ
}

DISTANCE_BETWEEN_CAMERAS = 0.12  # м (в метрах)
K = np.array(
    [
        [653.2621743, 0., 304.81329016],
        [0., 653.7921645, 229.71279112],
        [0., 0., 1.]
    ]
)

# Левая камера - базовая, поэтому ее P - это [I | 0]
P_LEFT = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
P_RIGHT = K @ np.hstack((np.eye(3), np.array([[DISTANCE_BETWEEN_CAMERAS], [0], [0]])))


# Инициализация ORB детектора
KEY_POINTS_QUANTITY = 500
ORB_DETECTOR = cv2.ORB_create(nfeatures=KEY_POINTS_QUANTITY)  # noqa

# Инициализация мэтчера ключевых точек
BF_MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

LEFT_CAMERA = cv2.VideoCapture(0)
RIGHT_CAMERA = cv2.VideoCapture(2)

if not LEFT_CAMERA.isOpened() or not RIGHT_CAMERA.isOpened():
    logger.error("Не удалось открыть одну из камер")

counter = 0

# Инициализация предыдущего состояния
previous_ret_left, previous_left_frame = LEFT_CAMERA.read()
previous_ret_right, previous_right_frame = RIGHT_CAMERA.read()

while True:
    ret_left, left_frame = LEFT_CAMERA.read()
    ret_right, right_frame = RIGHT_CAMERA.read()

    if not ret_left or not ret_right:
        logger.error("Не удалось захватить кадры с обеих камер")
        # нужно ли менять состояние????
        # previous_left_frame, previous_right_frame = left_frame, right_frame
        continue

    # keypoints - это объект, содержащий координаты, угол, масштаб
    # descriptors = это матрица с кол-вом строк, соответствующим кол-ву КЧ

    previous_keypoints_left, previous_descriptors_left = ORB_DETECTOR.detectAndCompute(previous_left_frame, None)
    previous_keypoints_right, previous_descriptors_right = ORB_DETECTOR.detectAndCompute(previous_right_frame, None)

    keypoints_left, descriptors_left = ORB_DETECTOR.detectAndCompute(left_frame, None)
    keypoints_right, descriptors_right = ORB_DETECTOR.detectAndCompute(right_frame, None)

    if keypoints_is_empty(keypoints_left, keypoints_right, descriptors_left, descriptors_right):
        logger.warning("Не найдены ключевые точки")
        # нужно ли менять состояние????
        previous_left_frame, previous_right_frame = left_frame, right_frame
        continue

    # Поиск соответствий
    previous_matches = BF_MATCHER.match(previous_descriptors_left, previous_descriptors_right)
    matches = BF_MATCHER.match(descriptors_left, descriptors_right)

    # Сортировка по расстоянию (чем меньше расстояние, тем лучше соответствие)
    previous_sorted_matches = sorted(previous_matches, key=lambda x: x.distance)
    sorted_matches = sorted(matches, key=lambda x: x.distance)

    # Соответствия между левыми кадрами для поиска 3D соответствий между
    # текущей и предыдущей стереопарами
    previous_left_and_current_left_matches = BF_MATCHER.match(previous_descriptors_left, descriptors_left)


    # Вызов функции триангуляции после поиска соответствий
    previous_points_3D = triangulate_points(
        P_LEFT, P_RIGHT, previous_keypoints_left, previous_keypoints_right, previous_sorted_matches
    )
    points_3D = triangulate_points(
        P_LEFT, P_RIGHT, keypoints_left, keypoints_right, sorted_matches
    )


    # Мапинг 3D-точек к их соответствиям
    previous_points_3D_map = {match.queryIndex: previous_points_3D[i] for i, match in enumerate(previous_sorted_matches)}
    points_3D_map = {match.queryIndex: points_3D[i] for i, match in enumerate(sorted_matches)}




    if previous_points_3D.size > 0 and points_3D.size > 0:
        R, t = estimate_motion(previous_points_3D, points_3D)
        R = np.round(R, decimals=3)
        t = np.round(t, decimals=3)
        logger.info(f"Изменение позы:\nМатрица вращения:\n{R}\nВектор трансляции:\n{t}")
    else:
        logger.warning("Не удалось вычислить 3D-точки для оценки движения")

    logger.info(f"Counter: {counter}")
    counter += 1

    # Смена состояния перед следующей итерацией
    previous_left_frame, previous_right_frame = left_frame, right_frame

    # Ожидаем нажатия клавиши 'q' для выхода
    if cv2.waitKey(1) & 0xFF == ord('q') or counter == 3:
        break

LEFT_CAMERA.release()
RIGHT_CAMERA.release()
cv2.destroyAllWindows()

# Для отладки вывод ключевых точек
# print_keypoints(keypoints_left, quantity=5)

# Для отладки отрисовка с ключевыми точками
# draw_with_keypoints(left_frame, right_frame, keypoints_left, keypoints_right)

# Для отладки отрисовка соответствий ключевых точек
# draw_keypoints_matches(left_frame, right_frame, keypoints_left, keypoints_right, sorted_matches)