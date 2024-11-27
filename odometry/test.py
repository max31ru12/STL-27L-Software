import cv2
import numpy as np

# Инициализация стереокамер
left_camera = cv2.VideoCapture(0)
right_camera = cv2.VideoCapture(2)

if not left_camera.isOpened() or not right_camera.isOpened():
    print("Не удалось открыть одну или обе камеры")
    exit()

# Настройка параметров SGBM для расчета диспаритета
SGBM_PARAMS = {
    "minDisparity": 0,
    "numDisparities": 16 * 5,  # Кратное 16
    "blockSize": 15,
    "uniquenessRatio": 10,
    "speckleWindowSize": 100,
    "speckleRange": 32,
    "disp12MaxDiff": 1,
    "preFilterCap": 63,
    "P1": 8 * 15 ** 2,
    "P2": 32 * 15 ** 2,
    "mode": cv2.STEREO_SGBM_MODE_SGBM_3WAY,
}
disparity_engine = cv2.StereoSGBM_create(**SGBM_PARAMS)

# Настройка детектора ключевых точек
fast_detector = cv2.FastFeatureDetector_create()

# Параметры для Лукаса-Канаде
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Переменные для накопления движения
cumulative_translation = np.zeros((2,))
cumulative_rotation = np.zeros((2, 2))  # 2x2 для вращения

# Главный цикл обработки
prev_left_frame = None
prev_points = None

while True:
    # Чтение кадров с камер
    ret_left, frame_left = left_camera.read()
    ret_right, frame_right = right_camera.read()

    if not ret_left or not ret_right:
        print("Не удалось захватить кадры с обеих камер")
        break

    # Преобразование в оттенки серого
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    # Вычисление карты диспаритета
    disparity = disparity_engine.compute(gray_left, gray_right).astype(np.float32) / 16.0
    cv2.imshow("Disparity", disparity / disparity.max())

    # Определение ключевых точек
    if prev_left_frame is None:
        prev_left_frame = gray_left
        prev_points = cv2.goodFeaturesToTrack(gray_left, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7)
        continue

    # Трекинг ключевых точек методом Лукаса-Канаде
    curr_points, st, err = cv2.calcOpticalFlowPyrLK(prev_left_frame, gray_left, prev_points, None, **lk_params)

    # Фильтрация точек
    good_prev = prev_points[st == 1]
    good_curr = curr_points[st == 1]

    # Оценка движения с помощью матрицы преобразования
    if len(good_prev) >= 4:
        transformation_matrix, _ = cv2.estimateAffinePartial2D(good_prev, good_curr)
        if transformation_matrix is not None:
            translation = transformation_matrix[:, 2]
            rotation = transformation_matrix[:, :2]

            # Накопление трансформации
            cumulative_translation += translation
            cumulative_rotation += rotation

            print(f"Translation: {translation}, Rotation: \n{rotation}")

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

    # Выход по нажатию клавиши
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
left_camera.release()
right_camera.release()
cv2.destroyAllWindows()