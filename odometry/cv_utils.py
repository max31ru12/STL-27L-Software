from typing import Any, Sequence

import cv2
from cv2 import Mat
from numpy import ndarray, dtype, generic


def make_image_gray(
        image: Mat | ndarray[Any, dtype[generic]] | ndarray
) -> cv2.UMat | Mat | ndarray[Any, dtype[generic]] | ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def binarize_image(
        gray_image_to_binarize: cv2.UMat | Mat | ndarray[Any, dtype[generic]] | ndarray
) -> tuple[float, cv2.UMat]:
    """
    Returns:
    ret - ?
    thresh - это бинаризованное изображение
    """
    binarized_ret, binarized_thresh = cv2.threshold(gray_image_to_binarize, 150, 255, cv2.THRESH_BINARY)
    return binarized_ret, binarized_thresh


def detect_contours(
        binarized_thresh: cv2.UMat
) -> tuple[Sequence[cv2.UMat], cv2.UMat]:
    return cv2.findContours(image=binarized_thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)


def detect_edges(gray_image):
    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
    return edges


if __name__ == "__main__":
    import numpy as np
    import cv2

    photo = np.zeros((300, 300, 3), dtype="uint8")

    photo[:] = 255, 0, 0

    cv2.imshow("Photo", photo)
    cv2.waitKey(0)

