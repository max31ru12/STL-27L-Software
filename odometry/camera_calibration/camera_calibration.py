import cv2
import numpy as np
import glob

# Размеры шахматной доски
chessboard_size = (9, 6)  # количество внутренних углов шахматной доски
square_size = 22  # размер квадрата на шахматной доске (в мм)

# Подготовка 3D-точек для шахматной доски, например (0,0,0), (1,0,0), ..., (8,5,0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Массивы для хранения точек объекта и точек изображения для всех изображений
obj_points = []  # 3D-точки в реальном пространстве
img_points = []  # 2D-точки в пространстве изображения

# Загрузка всех изображений шахматной доски для калибровки
images = glob.glob('./*.jpg')  # замените на путь к вашим изображениям

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Найти углы шахматной доски
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # Если углы найдены, добавить точки объекта и точки изображения
    if ret:
        obj_points.append(objp)
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                         criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        img_points.append(corners)

        # Нарисовать углы шахматной доски
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('Chessboard corners', img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# Калибровка камеры
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

# Печать матрицы камеры и коэффициентов искажения
print("Camera matrix (K):\n", camera_matrix)
print("Distortion coefficients:\n", dist_coeffs)

# Начальные значения матрицы вращения и трансляции
R = np.eye(3)  # предположим, что камера находится в начальной ориентации
t = np.zeros((3, 1))  # начальное смещение равно нулю

# Вычисление матрицы проекции P = [K | 0] для начальной позиции
Rt = np.hstack((R, t))  # объединение R и t
P = camera_matrix @ Rt  # матрица проекции

# Запись в файл calib.txt
with open('test_calib.txt', 'w') as f:
    f.write(' '.join(map(lambda x: f"{x:.12e}", P.flatten())) + '\n')
    # Пример для другой камеры или позиции (если требуется)
    P_alternate = camera_matrix @ np.hstack((R, np.array([[-0.3861448], [0], [0]])))  # пример с другим смещением по X
    f.write(' '.join(map(lambda x: f"{x:.12e}", P_alternate.flatten())) + '\n')

print("Матрицы калибровки успешно записаны в calib.txt")