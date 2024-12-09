import cv2
import numpy as np
import time

from config import FPS, SGBM_engine, LK_PARAMS
from odometry.processing_utils import ProcessingUtils
from odometry_utils import DebugTools

# Калибровочные параметры камеры
focal_length = 551.25345092  # Фокусное расстояние в пикселях
baseline = 0.12  # Базисное расстояние между камерами в метрах
fx, fy, cx, cy = 551.25345092, 551.43716708, 303.15262187, 220.81193167
camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=np.float32)

# Инициализация стереокамер
left_camera = cv2.VideoCapture(0)
right_camera = cv2.VideoCapture(2)

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

while True:

    time.sleep(1 / FPS)

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

    # Определение ключевых точек
    if prev_left_frame is None:
        prev_left_frame = gray_left
        prev_points = cv2.goodFeaturesToTrack(gray_left, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7)
        prev_3d_points = curr_3d_points
        continue

    # Трекинг ключевых точек методом Лукаса-Канаде
    curr_points, st, err = cv2.calcOpticalFlowPyrLK(prev_left_frame, gray_left, prev_points, None, **LK_PARAMS)

    # Фильтрация точек
    good_prev = prev_points[st == 1]
    good_curr = curr_points[st == 1]

    # Извлечение 3D-точек
    prev_3d = prev_3d_points[good_prev[:, 1].astype(int), good_prev[:, 0].astype(int)]
    curr_2d = good_curr[:, 0:2]

    # Проверяем корректность данных
    if len(prev_3d) < 4 or len(curr_2d) < 4:
        print("Недостаточно точек для solvePnPRansac")
        prev_left_frame = gray_left
        prev_points = good_curr.reshape(-1, 1, 2)
        prev_3d_points = curr_3d_points
        continue

    prev_3d = np.asarray(prev_3d, dtype=np.float32)
    curr_2d = np.asarray(curr_2d, dtype=np.float32)

    # Оценка движения с помощью solvePnPRansac
    _, rvec, tvec, inliers = cv2.solvePnPRansac(prev_3d, curr_2d, camera_matrix, None)

    # Преобразуем вращение в матрицу
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    # Накопление трансляции
    cumulative_translation += tvec.ravel()

    # Добавление координат в траекторию
    trajectory_x.append(cumulative_translation[0])
    trajectory_z.append(cumulative_translation[2])

    # Печать текущей трансляции и накопленных координат
    print(f"Current Translation (X, Y, Z): {np.round(tvec.ravel(), decimals=3)}")
    print(
        f"Cumulative Coordinates: X={np.round(cumulative_translation[0], decimals=3)}, "
        f"Z={np.round(cumulative_translation[2], decimals=3)}"
    )
    print(f"Rotation Matrix: \n{np.round(rotation_matrix, decimals=6)}\n")

    # Визуализация трекинга ключевых точек
    for i, (new, old) in enumerate(zip(good_curr, good_prev)):
        a, b = new.ravel()
        c, d = old.ravel()
        cv2.line(frame_left, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        cv2.circle(frame_left, (int(a), int(b)), 5, (255, 0, 0), -1)

    cv2.imshow("Optical Flow", frame_left)

    # Обновление предыдущего кадра и точек
    prev_left_frame = gray_left
    prev_points = good_curr.reshape(-1, 1, 2)
    prev_3d_points = curr_3d_points

    # Выход по нажатию клавиши
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
left_camera.release()
right_camera.release()
cv2.destroyAllWindows()

DebugTools.visualize_path(trajectory_x, trajectory_z, x_lim=10, y_lim=10)
