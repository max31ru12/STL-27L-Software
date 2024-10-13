from typing import Any, Sequence

import numpy as np
import cv2
from cv2 import Mat
from numpy import ndarray, dtype, generic

from odometry.cv_utils import make_image_gray, detect_contours, binarize_image, detect_edges

capture = cv2.VideoCapture(0)


while True:
    # ret: bool - удачно ли захвачено изображение
    ret, img = capture.read()

    gray_image = make_image_gray(img)

    lined_img = cv2.Canny(gray_image, 90, 90)

    cv2.imshow("Lined", lined_img)


    # Отрисовка контуров
    # _, thresh = binarize_image(gray_image)
    #
    # contours, hierarchy = detect_contours(thresh)
    # cv2.drawContours(
    #     image=img,
    #     contours=contours,
    #     contourIdx=-1,
    #     color=(0, 255, 0),
    #     thickness=1,
    #     lineType=cv2.LINE_AA
    # )


    # cv2.waitKey(50)
    # cv2.imwrite('contours_none_image1.jpg', img)
    # img - трехмерная матрица с тремя значениями интенсивности [B, G, R]
    # img[1, 1] - обращение к пикселю 1, 1
    # ширина и высота изображения, channels - кол-во цветовых каналов (RGB)
    # height, width = gray_image.shape
    # cv2.imshow('Video', gray_image)
    # print(f"iteration:: {gray_image[1, 1]} size: {gray_image.size} width: {width}, height: {height}")
    # print(f"intensity mean: {np.mean(gray_image)}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
