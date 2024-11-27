import sys
from typing import Iterable

import numpy as np
import cv2
from loguru import logger
import matplotlib.pyplot as plt


logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")


def undistort_image(
        img: cv2.Mat,
        camera_matrix: np.array,
        optimized_camera_matrix: np.array,
        distortion_coefficients,
        roi: Iterable,
):
    dst = cv2.undistort(img, camera_matrix, distortion_coefficients, None, optimized_camera_matrix)
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    return dst


def gaussian_filter_image(image, kernel_size: int = 7):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=0)


def read_images(left_camera, right_camera, gray=False):
    ret_left, left_frame = left_camera.read()
    ret_right, right_frame = right_camera.read()
    if gray:
        left_frame = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        right_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
    return ret_left, left_frame, ret_right, right_frame


class DebugTools:

    @classmethod
    def show_two_frames(cls, left, right) -> None:
        combined_frame = np.hstack((left, right))
        cv2.imshow("Left and right frames", combined_frame)

    @classmethod
    def show_disparity_map(cls, disparity, normalized=False) -> None:
        """
        Светлые области — ближние объекты.
        Тёмные области — дальние объекты.
        """
        if normalized:
            # нормализация нужна для визуализации
            disparity_normalized = cv2.normalize(disparity,  # noqa
                                                 None,
                                                 alpha=0,
                                                 beta=255,
                                                 norm_type=cv2.NORM_MINMAX,
                                                 dtype=cv2.CV_8U)
            cv2.imshow("Disparity Map", disparity_normalized)
        else:
            cv2.imshow("Disparity Map", disparity)

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
    def plot_3D_points(cls, previous_points_3D, points_3D):
        """
        Построение 3D облаков точек с одинаковым масштабом и системой координат.
        """
        # Находим общий диапазон координат для всех осей
        all_points = np.vstack((previous_points_3D, points_3D))
        x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
        y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
        z_min, z_max = max(0, np.min(all_points[:, 2])), np.max(all_points[:, 2])  # Z не должен быть отрицательным

        # Создаём канву с двумя подграфиками
        fig = plt.figure(figsize=(12, 6))

        # Первый график для previous_points_3D
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(previous_points_3D[:, 0], previous_points_3D[:, 2], previous_points_3D[:, 1], c='r', marker='o')
        ax1.set_title("Previous Points 3D")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Z")
        ax1.set_zlabel("Y")

        # Устанавливаем одинаковый масштаб
        ax1.set_xlim([x_min, x_max])
        ax1.set_ylim([z_min, z_max])
        ax1.set_zlim([y_min, y_max])

        # Второй график для points_3D
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(points_3D[:, 0], points_3D[:, 2], points_3D[:, 1], c='b', marker='^')
        ax2.set_title("Current Points 3D")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Z")
        ax2.set_zlabel("Y")

        # Устанавливаем одинаковый масштаб
        ax2.set_xlim([x_min, x_max])
        ax2.set_ylim([z_min, z_max])
        ax2.set_zlim([y_min, y_max])

        # Показать графики
        plt.tight_layout()
        plt.show()

    @classmethod
    def plot_2D_points_X_Z(cls, previous_points_3D, points_3D):
        # Находим общий диапазон координат для всех осей
        x_coordinates_current = points_3D[:, 0]
        z_coordinates_current = points_3D[:, 2]

        x_coordinates_previous = previous_points_3D[:, 0]
        z_coordinates_previous = previous_points_3D[:, 2]

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].scatter(x_coordinates_current, z_coordinates_current)
        axes[0].set_title("Текущие координаты")
        axes[0].set_xlabel("X")
        axes[0].set_ylabel("Z")
        axes[0].set_xlim(-5, 5)
        axes[0].set_ylim(0, 10)

        axes[1].scatter(x_coordinates_previous, z_coordinates_previous)
        axes[1].set_title("предыдущие координаты")
        axes[1].set_xlabel("X")
        axes[1].set_ylabel("Z")
        axes[1].set_xlim(-5, 5)
        axes[1].set_ylim(0, 10)

        plt.tight_layout()
        plt.show()


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
    # DON'T UPDATE PREVIOUS STATE
    return not KP_left or not KP_right or DES_left is None or DES_right is None


def triangulate_points(
        P_left: np.ndarray,
        P_right: np.ndarray,
        keypoints_left: list[cv2.KeyPoint],
        keypoints_right: list[cv2.KeyPoint],
        matches: list[cv2.DMatch],
        log=False,
) -> np.ndarray:
    """
    Выполняет триангуляцию для восстановления 3D-координат точек на основе совпадений.
    Нормализация не требуется.
    """

    # Функция для фильтрации выбросов
    def filter_outliers(points_3D):  # noqa
        mean = np.mean(points_3D, axis=0)
        std = np.std(points_3D, axis=0)
        mask = np.all(np.abs(points_3D - mean) < 3 * std, axis=1)
        return points_3D[mask]

    try:
        # Извлечение координат ключевых точек только для совпадений
        pts_left = np.array([keypoints_left[match.queryIdx].pt for match in matches], dtype=np.float32)
        pts_right = np.array([keypoints_right[match.trainIdx].pt for match in matches], dtype=np.float32)

        # Выполняем триангуляцию (Сделать гомогенную матрицу)
        points_4D_homogeneous = cv2.triangulatePoints(
            P_left, P_right,
            pts_left.T, pts_right.T
        )

        # Конвертация однородных координат в 3D (делим на w)
        points_3D = points_4D_homogeneous[:3] / points_4D_homogeneous[3]
        points_3D = points_3D.T  # Транспонируем для формата Nx3

        # Фильтрация выбросов
        # points_3D = filter_outliers(points_3D)

        # Удаляем точки с Z <= 0
        # valid_points_mask = points_3D[:, 2] > 0
        # points_3D = points_3D[valid_points_mask]


        if log:
            logger.info(f"Количество точек до фильтрации: {points_4D_homogeneous.shape[1]}")
            logger.info(f"Количество точек после фильтрации выбросов: {len(points_3D)}")
            logger.info(f"Диапазон Z-координат: min = {np.min(points_3D[:, 2])}, max = {np.max(points_3D[:, 2])}")

        if len(points_3D) == 0:
            logger.warning("Все 3D точки некорректны (Z <= 0).")
            return np.array([])

        return points_3D

    except cv2.error as e:
        logger.error(f"Ошибка OpenCV при триангуляции: {e}")
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
        log=False
) -> [np.ndarray, np.ndarray]:
    """
    Выходные данные имеют одинаковый размер
    """

    # Мапинг 3D-точек к их соответствиям с проверкой на выход за пределы
    previous_points_3D_map = {
        match.queryIdx: previous_keypoints_3D[match.queryIdx]
        for match in previous_matches
        if match.queryIdx < len(previous_keypoints_3D)
    }
    points_3D_map = {
        match.queryIdx: current_keypoints_3D[match.queryIdx]
        for match in matches
        if match.queryIdx < len(current_keypoints_3D)
    }

    previous_points_3D = []
    current_points_3D = []

    for match in previous_left_and_current_left_matches:
        prev_idx = match.queryIdx  # индекс точки на предыдущем левом изображении
        curr_idx = match.trainIdx  # индекс точки на текущем левом изображении

        # Проверяем, есть ли соответствующие 3D-точки в словарях
        if prev_idx in previous_points_3D_map and curr_idx in points_3D_map:
            previous_points_3D.append(previous_points_3D_map[prev_idx])
            current_points_3D.append(points_3D_map[curr_idx])

    # Преобразование списков в numpy массивы
    previous_points_3D = np.array(previous_points_3D)
    current_points_3D = np.array(current_points_3D)

    if log:
        logger.info(f"{len(previous_matches)=}, {len(matches)=}")
        if len(previous_points_3D) == 0 or len(current_points_3D) == 0:
            # Логирование для отладки
            logger.warning("Не удалось найти совпадающие 3D-точки между предыдущим и текущим кадрами.")

    return previous_points_3D, current_points_3D


def estimate_motion(previous_points_3D: np.ndarray, current_points_3D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Определяет изменение позы камеры между двумя наборами 3D-точек.
    """

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



