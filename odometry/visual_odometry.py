import cv2
import numpy as np

# Загрузка двух последовательных кадров
frame1 = cv2.imread('0.jpg')
frame2 = cv2.imread('5.jpg')

# Преобразование в градации серого
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

# Вычисление оптического потока
flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

# Вычисление среднего смещения по вертикали
average_flow_y = np.mean(flow[:, :, 1])  # Индекс 1 соответствует вертикальной компоненте потока

# Преобразование оптического потока в миллиметры
focal_length_mm = 29  # Фокусное расстояние камеры iPhone XR в миллиметрах
pixel_size_mm = 0.001  # Размеры пикселей в миллиметрах

displacement_mm = average_flow_y * pixel_size_mm * focal_length_mm

print("Смещение камеры в миллиметрах (вдоль оси Y):", displacement_mm)