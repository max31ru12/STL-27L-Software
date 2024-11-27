import sys

import numpy as np
import cv2
from loguru import logger
import matplotlib.pyplot as plt


logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")


def gaussian_filter_image(image, kernel_size: int = 7):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=0)


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
            cv2.imshow("Disparity Map", disparity / disparity.max())
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
