import sys

import numpy as np
import cv2

from loguru import logger

# DEBUG TOOLS

logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")


def draw_with_keypoints(left_frame, right_frame, keypoints_left, keypoints_right) -> None:  # noqa
    left_with_keypoints = cv2.drawKeypoints(left_frame, keypoints_left, None, color=(0, 255, 0), flags=0)
    right_with_keypoints = cv2.drawKeypoints(right_frame, keypoints_right, None, color=(0, 255, 0), flags=0)

    # Отображение кадров с ключевыми точками
    cv2.imshow("Left Camera with Keypoints", left_with_keypoints)
    cv2.imshow("Right Camera with Keypoints", right_with_keypoints)


def print_keypoints(keypoints, quantity: int | None = None):
    if quantity is None or quantity > len(matches):
        quantity = len(keypoints)
    for i, kp in enumerate(keypoints[:quantity]):  # Выводим первые 5 ключевых точек
        print(f"Точка {i + 1}: x={kp.pt[0]:.2f}, y={kp.pt[1]:.2f}, угол={kp.angle:.2f}, масштаб={kp.size:.2f}")


def keypoints_is_empty(KP_left, KP_right, DES_left, DES_right) -> bool:
    """
    KP_left - keypoints_left
    KP_right - keypoints_right
    DES_left - descriptors_left
    DES_right - descriptors_right
    """
    return not KP_left or not KP_right or DES_left is None or DES_right is None


def draw_keypoints_matches(left_frame, right_frame, KP_left, KP_right, matches, quantity: int | None = None):  # noqa
    """
    KP_left - keypoints_left
    KP_right - keypoints_right
    """
    if quantity is None or quantity > len(matches):
        quantity = len(matches)
    matched_frame = cv2.drawMatches(  # noqa
        left_frame,
        keypoints_left,
        right_frame,
        keypoints_right,
        matches[:quantity],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imshow("Matches", matched_frame)


### MAIN CODE  # noqa

# Калибровочные параметры камеры: (K - матрица калибровкиб P - )
# Формула проецирования точки x2d = Px3d
CAMERA_CALIBRATION_PARAMETERS = {
    "K_LEFT": np.array([[653.26, 0., 304.81],
                        [0., 653.79, 229.71],
                        [0., 0., 1.]]),
    "P_LEFT": np.array([[653.26, 0., 304.81],
                        [0., 653.79, 229.71],
                        [0., 0., 1.]]),
    "K_RIGHT": np.array([[653.26, 0., 304.81, 0.],
                         [0., 653.79, 229.71, 0.],
                         [0., 0., 1., 0.]]),
    "P_RIGHT": np.array([[6.5326e+02, 0.0000e+00, 3.0481e+02, -2.5200e-01],
                         [0.0000e+00, 6.5379e+02, 2.2971e+02, 0.0000e+00],
                         [0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00]]),
    "FOCUS_DISTANCE": 3.67,  # в мм (миллиметр),
    "EXTENSION": 3,  # в МегаПикселях
    "PIXEL_REAL_DIMENSIONS": 1.4,  # в мкм (микрометр) УТОЧНИТЬ
}

KEY_POINTS_QUANTITY = 500

# Инициализация ORB детектора
ORB_DETECTOR = cv2.ORB_create(nfeatures=KEY_POINTS_QUANTITY)  # noqa
BF_MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

LEFT_CAMERA = cv2.VideoCapture(0)
RIGHT_CAMERA = cv2.VideoCapture(2)

if not LEFT_CAMERA.isOpened() or not RIGHT_CAMERA.isOpened():
    logger.error("Не удалось открыть одну из камер")

counter = 0

while True:
    ret_left, left_frame = LEFT_CAMERA.read()
    ret_right, right_frame = RIGHT_CAMERA.read()

    if not ret_left or not ret_right:
        logger.error("Не удалось захватить кадры с обеих камер")
        continue

    # keypoints - это объект, содержащаий координаты, угол, масштаб
    # descriptors = это матрица с кол-вом строк, соответствующим кол-ву КЧ
    keypoints_left, descriptors_left = ORB_DETECTOR.detectAndCompute(left_frame, None)
    keypoints_right, descriptors_right = ORB_DETECTOR.detectAndCompute(right_frame, None)

    if keypoints_is_empty(keypoints_left, keypoints_right, descriptors_left, descriptors_left):
        logger.warning("Не найдены ключевые точки")
        continue

    # Поиск соответствий
    matches = BF_MATCHER.match(descriptors_left, descriptors_right)
    # Сортировка по расстоянию (чем меньше расстояние, тем лучше соответствие)
    sorted_matches = sorted(matches, key=lambda x: x.distance)

    # Для отладки вывод ключевых точек
    # print_keypoints(keypoints_left, quantity=5)

    # Для отладки отрисовка с ключевыми точками
    # draw_with_keypoints(left_frame, right_frame, keypoints_left, keypoints_right)

    # Для отладки отрисовка соответствий ключевых точек
    # draw_keypoints_matches(left_frame, right_frame, keypoints_left, keypoints_right, sorted_matches)

    print(counter)
    counter += 1

    # Ожидаем нажатия клавиши 'q' для выхода
    if cv2.waitKey(1) & 0xFF == ord('q') or counter == 300:
        break

LEFT_CAMERA.release()
RIGHT_CAMERA.release()
cv2.destroyAllWindows()
