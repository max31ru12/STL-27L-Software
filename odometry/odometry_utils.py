import sys

import numpy as np
import cv2
from loguru import logger


logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")


def draw_with_keypoints(
        left_frame: cv2.Mat,
        right_frame: cv2.Mat,
        keypoints_left: list[cv2.KeyPoint],
        keypoints_right: list[cv2.KeyPoint],
) -> None:  # noqa
    left_with_keypoints = cv2.drawKeypoints(left_frame, keypoints_left, None, color=(0, 255, 0), flags=0)  # noqa
    right_with_keypoints = cv2.drawKeypoints(right_frame, keypoints_right, None, color=(0, 255, 0), flags=0)  # noqa

    # Отображение кадров с ключевыми точками
    cv2.imshow("Left Camera with Keypoints", left_with_keypoints)
    cv2.imshow("Right Camera with Keypoints", right_with_keypoints)


def print_keypoints(keypoints: list[cv2.KeyPoint], quantity: int | None = None):
    if quantity is None or quantity > len(keypoints):
        quantity = len(keypoints)
    for i, kp in enumerate(keypoints[:quantity]):  # Выводим первые 5 ключевых точек
        print(f"Точка {i + 1}: x={kp.pt[0]:.2f}, y={kp.pt[1]:.2f}, угол={kp.angle:.2f}, масштаб={kp.size:.2f}")


def keypoints_is_empty(
        KP_left: list[cv2.KeyPoint],
        KP_right: list[cv2.KeyPoint],
        DES_left: cv2.Mat | None,
        DES_right: cv2.Mat | None,
) -> bool:
    """
    KP_left - keypoints_left
    KP_right - keypoints_right
    DES_left - descriptors_left
    DES_right - descriptors_right
    """
    return not KP_left or not KP_right or DES_left is None or DES_right is None


def draw_keypoints_matches(
        left_frame: cv2.UMat,
        right_frame: cv2.UMat,
        KP_left: list[cv2.KeyPoint],
        KP_right: list[cv2.KeyPoint],
        matches: list[cv2.DMatch],
        quantity: int | None = None
):  # noqa
    """
    KP_left - keypoints_left
    KP_right - keypoints_right
    """
    if quantity is None or quantity > len(matches):
        quantity = len(matches)
    matched_frame = cv2.drawMatches(  # noqa
        left_frame,
        KP_left,
        right_frame,
        KP_right,
        matches[:quantity],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imshow("Matches", matched_frame)


def print_matches(matches: list[cv2.DMatch], quantity: int | None = None):
    if quantity is None or quantity > len(matches):
        quantity = len(matches)
    for match in matches[:quantity]:  # noqa
        print(
            f"Index in Left Image: {match.queryIdx}, Index in Right Image: {match.trainIdx}, Distance: {match.distance}"
        )


def triangulate_points(
        P_left: np.ndarray,
        P_right: np.ndarray,
        keypoints_left: list[cv2.KeyPoint],
        keypoints_right: list[cv2.KeyPoint],
        matches: list[cv2.DMatch]
) -> np.ndarray:
    """
    Выполняет триангуляцию для восстановления 3D-координат точек на основе совпадений.

    Параметры:
    - P_left: np.ndarray - Проекционная матрица для левой камеры (размер 3x4).
    - P_right: np.ndarray - Проекционная матрица для правой камеры (размер 3x4).
    - keypoints_left: list[cv2.KeyPoint] - Ключевые точки на левом изображении.
    - keypoints_right: list[cv2.KeyPoint] - Ключевые точки на правом изображении.
    - matches: list[cv2.DMatch] - Соответствия ключевых точек между левым и правым изображениями.

    Возвращает:
    - np.ndarray - Массив 3D-координат точек (размер Nx3), где N — количество точек.
    """
    try:
        # Извлечение координат ключевых точек только для совпадений
        pts_left = np.array([keypoints_left[match.queryIdx].pt for match in matches], dtype=np.float32)
        pts_right = np.array([keypoints_right[match.trainIdx].pt for match in matches], dtype=np.float32)

        # Выполняем триангуляцию
        points_4D_homogeneous = cv2.triangulatePoints(P_left, P_right, pts_left.T, pts_right.T)

        # Конвертация однородных координат в 3D (делим на w)
        points_3D = points_4D_homogeneous[:3] / points_4D_homogeneous[3]
        points_3D = points_3D.T  # Транспонируем для получения формата Nx3

        logger.info("Успешно выполнена триангуляция.")
        return points_3D

    except cv2.error as e:
        logger.error(f"Ошибка OpenCV при триангуляции: {e}")
        return np.array([])  # Возвращаем пустой массив, если произошла ошибка OpenCV

    except ValueError as e:
        logger.error(f"Ошибка значения: {e}")
        return np.array([])

    except Exception as e:
        logger.error(f"Неизвестная ошибка: {e}")
        return np.array([])


if __name__ == "__main__":
    import numpy as np
    import cv2

    photo = np.zeros((300, 300, 3), dtype="uint8")

    photo[:] = 255, 0, 0

    cv2.imshow("Photo", photo)
    cv2.waitKey(0)

