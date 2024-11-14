import sys
from typing import Any

import numpy as np
import cv2
from loguru import logger
from matplotlib import pyplot as plt

logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")


class DebugTools:

    @classmethod
    def draw_with_keypoints(
            cls,
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

    @classmethod
    def print_keypoints(cls, keypoints: list[cv2.KeyPoint], quantity: int | None = None) -> None:
        if quantity is None or quantity > len(keypoints):
            quantity = len(keypoints)
        for i, kp in enumerate(keypoints[:quantity]):  # Выводим первые 5 ключевых точек
            print(f"Точка {i + 1}: x={kp.pt[0]:.2f}, y={kp.pt[1]:.2f}, угол={kp.angle:.2f}, масштаб={kp.size:.2f}")

    @classmethod
    def draw_keypoints_matches(
            cls,
            left_frame: cv2.UMat,
            right_frame: cv2.UMat,
            KP_left: list[cv2.KeyPoint],
            KP_right: list[cv2.KeyPoint],
            matches: list[cv2.DMatch],
            quantity: int | None = None
    ) -> None:  # noqa
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

    @classmethod
    def print_matches(cls, matches: list[cv2.DMatch], quantity: int | None = None) -> None:
        if quantity is None or quantity > len(matches):
            quantity = len(matches)
        for match in matches[:quantity]:  # noqa
            print(
                f"Index in Left Image: {match.queryIdx}, Index in Right Image: {match.trainIdx}, Distance: {match.distance}"
            )

    @classmethod
    def visualize_path(cls, estimated_points_x, estimated_points_z, x_lim: int = 100, y_lim: int = 100) -> None:
        # Разбиваем estimated_path на X и Z координаты
        x_coords = estimated_points_x
        z_coords = estimated_points_z

        # Создаём график
        plt.figure(figsize=(10, 6))
        plt.plot(x_coords, z_coords, marker='o', markersize=3, color="blue", linewidth=1, label="Estimated Path")

        # Подписи для графика
        plt.xlabel("X Position")
        plt.ylabel("Z Position")
        plt.title("Estimated Path Visualization")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')  # Чтобы сохранить пропорции
        plt.xlim(-x_lim, x_lim)
        plt.ylim(-y_lim, y_lim)
        plt.show()


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

        # logger.info("Успешно выполнена триангуляция.")
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


def get_matched_3D_points(
        previous_keypoints_3D,
        current_keypoints_3D,
        previous_matches,
        matches,
        previous_left_and_current_left_matches,
):
    # Мапинг 3D-точек к их соответствиям
    previous_points_3D_map = {match.queryIdx: previous_keypoints_3D[i] for i, match in enumerate(previous_matches)}
    points_3D_map = {match.queryIdx: current_keypoints_3D[i] for i, match in enumerate(matches)}

    matched_3D_points = []

    for match in previous_left_and_current_left_matches:
        prev_idx = match.queryIdx  # индекс точки на предыдущем левом изображении
        curr_idx = match.trainIdx  # индекс точки на текущем левом изображении

        # Проверяем, есть ли соответствующие 3D-точки в словарях
        if prev_idx in previous_points_3D_map and curr_idx in points_3D_map:
            matched_3D_points.append((previous_points_3D_map[prev_idx], points_3D_map[curr_idx]))


def filter_matched_points(previous_points_3D, current_points_3D):
    """
    Фильтрует 3D-точки, чтобы количество точек в двух кадрах совпадало.
    """
    min_len = min(len(previous_points_3D), len(current_points_3D))
    return previous_points_3D[:min_len], current_points_3D[:min_len]


def estimate_motion(previous_points_3D: np.ndarray, current_points_3D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Определяет изменение позы камеры между двумя наборами 3D-точек.
    """
    # Фильтрация точек для совпадения по размеру
    previous_points_3D, current_points_3D = filter_matched_points(previous_points_3D, current_points_3D)

    if previous_points_3D.shape[0] < 4 or current_points_3D.shape[0] < 4:
        logger.warning("Недостаточно 3D-точек для оценки движения.")
        return np.eye(3), np.zeros((3, 1))

    try:
        # Используем метод Umeyama для оценки R и t
        prev_centered = previous_points_3D - previous_points_3D.mean(axis=0)
        curr_centered = current_points_3D - current_points_3D.mean(axis=0)

        H = np.dot(curr_centered.T, prev_centered)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(U, Vt)

        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = np.dot(U, Vt)

        t = current_points_3D.mean(axis=0) - R @ previous_points_3D.mean(axis=0)

        return R, t.reshape(-1, 1)

    except Exception as e:
        logger.error(f"Ошибка при оценке движения: {e}")
        return np.eye(3), np.zeros((3, 1))



