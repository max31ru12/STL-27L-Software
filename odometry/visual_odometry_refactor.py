import sys

import cv2
import numpy as np
from loguru import logger

from config import SGBM_engine, LK_PARAMS
from odometry.processing_utils import ProcessingUtils, Filters
from odometry_utils import DebugTools

logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")

# Калибровочные параметры камеры
focal_length = 551.25345092  # Фокусное расстояние в пикселях
baseline = 0.12  # Базисное расстояние между камерами в метрах
fx, fy, cx, cy = 551.25345092, 551.43716708, 303.15262187, 220.81193167
camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=np.float32)
displacement_threshold = 1

# Инициализация стереокамер
left_camera = cv2.VideoCapture(0)
right_camera = cv2.VideoCapture(2)


# Параметры записи видео
output_file = "../slam/output.avi"  # Имя выходного файла
disparity_output_file = "optical_flow/disparity_output.avi"  # Имя выходного файла
frame_width = 640          # Ширина кадра
frame_height = 480         # Высота кадра
fps = left_camera.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Кодек (например, 'XVID' для AVI или 'MP4V' для MP4)

out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))


if not left_camera.isOpened() or not right_camera.isOpened():
    print("Не удалось открыть одну или обе камеры")
    exit()

# Списки для накопления координат
trajectory_x = [0]  # Начальное положение X
trajectory_z = [0]  # Начальное положение Z

# Главный цикл обработки
prev_left_frame = None
prev_points = None
prev_3d_points = None

# Начальные координаты
cumulative_translation = np.array([0.0, 0.0, 0.0])  # x, y, z

counter = 0

while True:

    try:
        ret_left, frame_left, ret_right, frame_right = ProcessingUtils.read_images(left_camera, right_camera)
        if not ret_left or not ret_right:
            print("Не удалось захватить кадры с обеих камер")
            break

        # Преобразование в оттенки серого
        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

        disparity = SGBM_engine.compute(gray_left, gray_right).astype(np.float32) / 16.0
        DebugTools.show_disparity_map(disparity, normalized=True)

        curr_3d_points = ProcessingUtils.disparity_to_3d(disparity, focal_length, baseline, cx, cy)

        # Определение ключевых точек (для первой итерации)
        if prev_left_frame is None:
            prev_left_frame = gray_left
            prev_points = cv2.goodFeaturesToTrack(gray_left, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7)
            prev_3d_points = curr_3d_points
            continue

        curr_points, st, err = cv2.calcOpticalFlowPyrLK(prev_left_frame, gray_left, prev_points, None, **LK_PARAMS)

        good_prev = prev_points[st == 1]
        good_curr = curr_points[st == 1]
        good_prev, good_curr = Filters.filter_points(good_prev, good_curr)

        prev_3d = prev_3d_points[good_prev[:, 1].astype(int), good_prev[:, 0].astype(int)]
        curr_2d = good_curr[:, 0:2]

        if len(prev_3d) < 4 or len(curr_2d) < 4:
            logger.warning("Недостаточно точек для solvePnPRansac. Повторная инициализация ключевых точек.")
            # Переинициализация ключевых точек
            prev_left_frame = gray_left
            prev_points = cv2.goodFeaturesToTrack(gray_left, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7)
            prev_3d_points = curr_3d_points
            counter += 1
            continue

        prev_3d = np.asarray(prev_3d, dtype=np.float32)
        curr_2d = np.asarray(curr_2d, dtype=np.float32)

        _, rvec, tvec, inliers = cv2.solvePnPRansac(prev_3d, curr_2d, camera_matrix, None)
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        cumulative_translation += tvec.ravel()

        if tvec is not None and np.isfinite(tvec).all():
            # Вычисление смещения относительно предыдущей точки
            displacement = np.round(tvec.ravel(), decimals=4)
            if np.any(np.abs(displacement) > displacement_threshold):
                logger.warning(f"Шумовое смещение обнаружено: {displacement}. Игнорируем его.")
            else:
                logger.info(displacement)
                new_x = trajectory_x[-1] + displacement[0]
                new_z = trajectory_z[-1] + displacement[2]
                trajectory_x.append(new_x)
                trajectory_z.append(new_z)

        # Визуализация трекинга ключевых точек
        for i, (new, old) in enumerate(zip(good_curr, good_prev)):
            a, b = new.ravel()
            c, d = old.ravel()
            cv2.line(frame_left, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            cv2.circle(frame_left, (int(a), int(b)), 5, (255, 0, 0), -1)

        cv2.imshow("Optical Flow", frame_left)
        cv2.imwrite(f"optical_flow/{counter}.jpeg", frame_left)
        out.write(frame_left)

        # Обновление предыдущего кадра и точек
        prev_left_frame = gray_left
        prev_points = good_curr.reshape(-1, 1, 2)
        prev_3d_points = curr_3d_points

    except KeyboardInterrupt as e:
        left_camera.release()
        right_camera.release()
        cv2.destroyAllWindows()
        DebugTools.visualize_path(trajectory_x, trajectory_z, x_lim=10, y_lim=10)
        raise e

    except Exception as e:
        logger.error(e)
        raise e

    counter += 1
    print(counter)

    if counter > 250:
        break

    # Выход по нажатию клавиши
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


left_camera.release()
right_camera.release()
cv2.destroyAllWindows()
DebugTools.visualize_path(trajectory_x, trajectory_z, x_lim=10, y_lim=10)
