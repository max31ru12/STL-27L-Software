import numpy as np
import cv2

capture = cv2.VideoCapture(0)

i = 0

while True:
    i += 1
    ret, img = capture.read()

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # img - трехмерная матрица с тремя значениями интенсивности [B, G, R]
    # img[1, 1] - обращение к пикселю 1, 1

    # ширина и высота изображения, channels - кол-во цветовых каналов (RGB)
    height, width = gray_image.shape

    cv2.imshow('Video', gray_image)

    print(f"iteration: {i}: {gray_image[1, 1]} size: {gray_image.size} width: {width}, height: {height}")
    print(f"intensity mean: {np.mean(gray_image)}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
