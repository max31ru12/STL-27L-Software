import cv2

# Открытие камеры (обычно 0 соответствует встроенной камере)
cap = cv2.VideoCapture(1)

# Проверка, удалось ли открыть камеру
if not cap.isOpened():
    print("Не удалось открыть камеру")
else:
    # Захват одного кадра
    ret, frame = cap.read()

    # Проверка, что кадр успешно получен
    if ret:
        # Отображение кадра
        cv2.imshow('Captured Frame', frame)

        # Сохранение кадра в файл
        cv2.imwrite('captured_frame_4.jpg', frame)

        # Ожидание нажатия клавиши для закрытия окна
        cv2.waitKey(0)
    else:
        print("Не удалось захватить кадр")

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()