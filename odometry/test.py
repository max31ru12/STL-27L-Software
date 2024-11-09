import numpy as np
import cv2

from pprint import pprint

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
ORB_DETECTOR = cv2.ORB_create(nfeatures=KEY_POINTS_QUANTITY)





LEFT_CAMERA = cv2.VideoCapture(0)
RIGHT_CAMERA = cv2.VideoCapture(2)

if not LEFT_CAMERA.isOpened() or not RIGHT_CAMERA.isOpened():
    print("Ошибка: не удалось открыть одну из камер")

while True:
    ret_left, left_frame = LEFT_CAMERA.read()
    ret_right, right_frame = RIGHT_CAMERA.read()

    if not ret_left or not ret_right:
        print("Ошибка: не удалось захватить кадры с обеих камер")
        continue

    cv2.imshow("Left Camera", left_frame)
    cv2.imshow("Right Camera", right_frame)

    # Ожидаем нажатия клавиши 'q' для выхода
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
