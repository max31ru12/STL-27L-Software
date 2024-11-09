import time
from os import PathLike
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

Frame = cv2.Mat | np.ndarray[Any, np.dtype[np.generic]] | np.ndarray


class StereoVisualOdometry:  # noqa

    lk_params: ClassVar = dict(winSize=(15, 15),
                               flags=cv2.MOTION_AFFINE,
                               maxLevel=3,
                               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))
    EARLY_TERMINATION_THRESHOLD: ClassVar[int] = 5

    def __init__(self, left_camera_number: int = 1, right_camera_number: int = 2):
        # Cameras
        self.previous_matches = [np.array([0.0, 0.0, 0.0]) for _ in range(4)]
        self.left_capture = cv2.VideoCapture(right_camera_number)
        self.right_capture = cv2.VideoCapture(left_camera_number)

        self.K_left, self.P_left = self.__load_calib("calib.txt")
        self.K_right, self.P_right = self.__load_calib("calib.txt")

        print(self.P_left)

        # CURRENT STATE
        self.left_current_frame: Frame | None = None
        self.right_current_frame: Frame | None = None

        self.ret_left = False
        self.ret_right = False

        # PREVIOUS STATE
        self.left_previous_frame = None
        self.right_previous_frame = None

        # что это такое?
        block = 11
        P1 = block * block * 8
        P2 = block * block * 32
        self.disparity = cv2.StereoSGBM_create(minDisparity=0, numDisparities=32, blockSize=block, P1=P1, P2=P2)  # noqa
        self.fast_features = cv2.FastFeatureDetector_create()  # noqa

    @property
    def current_frames_is_none(self) -> bool:
        return self.left_current_frame is None and self.right_current_frame is None

    @property
    def previous_frames_is_none(self) -> bool:
        return self.left_previous_frame is None and self.right_previous_frame is None

    @classmethod
    def visualize_path(cls, estimated_points, x_lim: int = 10, y_lim: int = 10):
        # Разбиваем estimated_path на X и Z координаты
        x_coords = [pos[0] for pos in estimated_points]
        z_coords = [pos[1] for pos in estimated_points]

        # Создаём график
        plt.figure(figsize=(10, 6))
        plt.plot(x_coords, z_coords, marker='o', markersize=3, color="blue", linewidth=1, label="Estimated Path")

        # Подписи для графика
        plt.xlabel("X Position")
        plt.ylabel("Z Position")
        plt.title("Estimated Path Visualization")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')  # Чтобы сохранить пропорции
        plt.xlim(-x_lim, x_lim)
        plt.ylim(-y_lim, y_lim)
        plt.show()

    @staticmethod
    def __load_calib(filepath: str | PathLike) -> tuple[np.ndarray, np.ndarray]:
        """
        loading camera calibration parameters
        """
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]
        return K, P

    @staticmethod
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    @staticmethod
    def calculate_right_qs(
            keypoints_img1,
            keypoints_img2,
            disparity1,
            disparity2,
            min_disparity: float = 0.0,
            max_disparity: float = 100.0
    ):
        """
        Возвращает:
        keypoints_img1_left: Отфильтрованные координаты ключевых точек на первом изображении.
        keypoints_img1_right: Соответствующие координаты ключевых точек на правом изображении (вычисленные путем сдвига на диспаритет).
        keypoints_img2_left: Отфильтрованные координаты ключевых точек на втором изображении.
        keypoints_img2_right: Соответствующие координаты ключевых точек на правом изображении (вычисленные путем сдвига на диспаритет).
        """

        # min_disp и max_disp - нужны для фильтрации шума или неправильных значений
        # Вычисляет диспаритет для заданных ключевых точек
        def get_idxs(q, disp):
            q_idx = q.astype(int)
            disp = disp.T[q_idx[:, 0], q_idx[:, 1]]
            return disp, np.where(np.logical_and(min_disparity < disp, disp < max_disparity), True, False)

        # Получаем диспаритеты и маски для фиотрации
        disparity_img1, mask1 = get_idxs(keypoints_img1, disparity1)
        disparity_img2, mask2 = get_idxs(keypoints_img2, disparity2)

        in_bounds = np.logical_and(mask1, mask2)

        # Фильтрует ключевые точки и значения диспаритета, оставляя только валидные точки
        keypoints_img1_left = keypoints_img1[in_bounds]
        keypoints_img2_left = keypoints_img2[in_bounds]
        disparity_img1 = disparity_img1[in_bounds]
        disparity_img2 = disparity_img2[in_bounds]

        # Создает копии фильтрованных ключевых точек, которые будут модифицированы для получения координат правых точек
        keypoints_img1_right = np.copy(keypoints_img1_left)
        keypoints_img2_right = np.copy(keypoints_img2_left)

        # Сдвигает координаты x на величину диспаритета, чтобы получить положение точки на правом изображении
        keypoints_img1_right[:, 0] -= disparity_img1
        keypoints_img2_right[:, 0] -= disparity_img2

        return keypoints_img1_left, keypoints_img1_right, keypoints_img2_left, keypoints_img2_right

    def read_images(self, gray=False, show=False):

        if self.current_frames_is_none and self.previous_frames_is_none:
            self.ret_left, self.left_current_frame = self.left_capture.read()
            self.ret_right, self.right_current_frame = self.right_capture.read()

        else:
            # make current state previous
            self.left_previous_frame = self.left_current_frame
            self.right_previous_frame = self.right_current_frame
            # get new state
            self.ret_left, self.left_current_frame = self.left_capture.read()
            self.ret_right, self.right_current_frame = self.right_capture.read()

        if gray:
            self.left_current_frame = cv2.cvtColor(self.left_current_frame, cv2.COLOR_BGR2GRAY)
            self.right_current_frame = cv2.cvtColor(self.right_current_frame, cv2.COLOR_BGR2GRAY)

        if show:
            cv2.imshow("Left", self.left_current_frame)
            cv2.imshow("Right", self.right_current_frame)
            cv2.waitKey(200)

    def reprojection_residuals(self, dof, q1, q2, Q1, Q2):
        """
        Calculate the residuals
        """
        # Get the rotation vector
        r = dof[:3]
        # Create the rotation matrix from the rotation vector
        R, _ = cv2.Rodrigues(r)
        # Get the translation vector
        t = dof[3:]
        # Create the transformation matrix from the rotation matrix and translation vector
        transf = self._form_transf(R, t)

        # Create the projection matrix for the i-1'th image and i'th image
        f_projection = np.matmul(self.P_left, transf)
        b_projection = np.matmul(self.P_left, np.linalg.inv(transf))

        # Make the 3D points homogenize
        ones = np.ones((q1.shape[0], 1))
        Q1 = np.hstack([Q1, ones])
        Q2 = np.hstack([Q2, ones])

        # Project 3D points from i'th image to i-1'th image
        q1_pred = Q2.dot(f_projection.T)
        # Un-homogenize
        q1_pred = q1_pred[:, :2].T / np.where(q1_pred[:, 2] == 0, 1e-10, q1_pred[:, 2])

        # Project 3D points from i-1'th image to i'th image
        q2_pred = Q1.dot(b_projection.T)
        # Un-homogenize
        q2_pred = q2_pred[:, :2].T / np.where(q2_pred[:, 2] == 0, 1e-10, q2_pred[:, 2])

        # Calculate the residuals
        residuals = np.vstack([q1_pred - q1.T, q2_pred - q2.T]).flatten()
        return residuals

    def get_tiled_keypoints(self, img, tile_h, tile_w):
        """
        Возвращает ключевые точки
        """

        def get_kps(x, y):
            # Выбираем плитку
            impatch = img[y:y + tile_h, x:x + tile_w]
            # Детектируем ключевые точки
            keypoints = self.fast_features.detect(impatch)
            # Correct the coordinate for the point
            for pt in keypoints:
                pt.pt = (pt.pt[0] + x, pt.pt[1] + y)

            # Get the 10 best keypoints
            if len(keypoints) > 10:
                keypoints = sorted(keypoints, key=lambda keypoint: -keypoint.response)
                return keypoints[:10]
            return keypoints

        # Высота и ширина изображения
        h, w, *_ = img.shape
        # Получаем ключевые точки для каждой из плиток
        kp_list = [get_kps(x, y) for y in range(0, h, tile_h) for x in range(0, w, tile_w)]
        # Преобразовать список ключевых точек в одномерный массив
        kp_list_flatten = np.concatenate(kp_list)
        return kp_list_flatten

    def track_keypoints(self, img1, img2, keypoints1, max_error: int = 4):
        """
        Принимает старое и новое изображение с одной камеры
        """

        # переводим в массив координат
        trackpoints1 = np.expand_dims(cv2.KeyPoint_convert(keypoints1), axis=1)
        # Отслеживаем ключевые точки с 1-го изображения на втором изображении
        trackpoints2, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, trackpoints1, None, **self.lk_params)

        # ФИЛЬТРЫ ДЛЯ ТОЧЕК
        trackable = st.astype(bool)  # массив успеха/неудачи для каждой точки
        under_thresh = np.where(err[trackable] < max_error, True, False)

        # Фильтруем точки
        trackpoints1 = trackpoints1[trackable][under_thresh]
        trackpoints2 = np.around(trackpoints2[trackable][under_thresh])

        h, w = img1.shape
        # Проверяет, находятся ли координаты trackpoints2 в пределах изображения,
        # сохраняя True для точек, которые находятся внутри границ.
        in_bounds = np.where(np.logical_and(trackpoints2[:, 1] < h, trackpoints2[:, 0] < w), True, False)

        # Еще одна фильтрация
        trackpoints1 = trackpoints1[in_bounds]
        trackpoints2 = trackpoints2[in_bounds]

        return trackpoints1, trackpoints2

    def calc_3d(
            self,
            keypoints_img1_left,
            keypoints_img1_right,
            keypoints_img2_left,
            keypoints_img2_right
    ):
        """
        Возвращает координаты 3D-точек для текущей стереопары и предыдущей стереопары
        """

        # триангуляция точек с предыщих изображений triangulate points from i-1'th image
        Q1 = cv2.triangulatePoints(self.P_left, self.P_right, keypoints_img1_left.T, keypoints_img1_right.T)

        Q1 = np.transpose(Q1[:3] / Q1[3])

        # триангуляция точек с текущих изображений triangulate points from i'th image
        Q2 = cv2.triangulatePoints(self.P_left, self.P_right, keypoints_img2_left.T, keypoints_img2_right.T)
        Q2 = np.transpose(Q2[:3] / Q2[3])
        return Q1, Q2

    def estimate_pose(self, q1, q2, Q1, Q2, max_iter=100):

        if q1.shape[0] < 6:
            print("Недостаточно совпадающих точек для оценки позы.")
            return np.eye(4)

        min_error = float('inf')
        early_termination = 0

        for _ in range(max_iter):

            # При отсутствии совпадений, не может выбрать из пустой матрицы ПОФИКСИТЬ
            sample_idx = np.random.choice(range(q1.shape[0]), 6)
            sample_q1, sample_q2, sample_Q1, sample_Q2 = q1[sample_idx], q2[sample_idx], Q1[sample_idx], Q2[sample_idx]
            opt_res = least_squares(self.reprojection_residuals, np.zeros(6), method="lm", max_nfev=200,
                                    args=(sample_q1, sample_q2, sample_Q1, sample_Q2))

            error = self.reprojection_residuals(opt_res.x, q1, q2, Q1, Q2).reshape((Q1.shape[0] * 2, 2))
            error = np.sum(np.linalg.norm(error, axis=1))

            if error < min_error:
                min_error = error
                out_pose = opt_res.x
                early_termination = 0
            else:
                early_termination += 1

            if early_termination == self.EARLY_TERMINATION_THRESHOLD:
                break

            r, t = out_pose[:3], out_pose[3:]  # noqa
            R, _ = cv2.Rodrigues(r)
            transformation_matrix = self._form_transf(R, t)  # noqa

            return transformation_matrix

    def get_pose(self, old_imgL, old_imgR, new_imgL, new_imgR):

        img1_l = old_imgL
        img2_l = new_imgL

        # Получаем ключевые точки
        left_keypoints1 = self.get_tiled_keypoints(img1_l, 10, 20)

        tp1_l, tp2_l = self.track_keypoints(img1_l, img2_l, left_keypoints1)

        old_disp = np.divide(self.disparity.compute(old_imgL, old_imgR).astype(np.float32), 16)
        new_disp = np.divide(self.disparity.compute(new_imgL, new_imgR).astype(np.float32), 16)

        tp1_l, tp1_r, tp2_l, tp2_r = self.calculate_right_qs(tp1_l, tp2_l, old_disp, new_disp)

        # Условие, проверяющее отсутствие совпадений
        try:
            Q1, Q2 = self.calc_3d(tp1_l, tp1_r, tp2_l, tp2_r)
            pass
        except Exception as e:
            print(e)
        # try:
        #     if tp1_l.size == 0 or tp2_l.size == 0 or tp1_r.size == 0 or tp2_r.size == 0:
        #         print("Не удалось обнаружить совпадающие точки. Пропуск текущего шага.")
        #         Q1, Q2 = self.calc_3d(*self.previous_matches)
        #     else:
        #         Q1, Q2 = self.calc_3d(tp1_l, tp1_r, tp2_l, tp2_r)
        # except Exception as e:
        #     print(tp1_l, tp1_r, tp2_l, tp2_r, sep="\n\n\n")
        #     # raise e

        self.previous_matches = [tp1_l, tp1_r, tp2_l, tp2_r]
        transformation_matrix = self.estimate_pose(tp1_l, tp2_l, Q1, Q2)  # noqa

        return transformation_matrix


if __name__ == "__main__":
    vo = StereoVisualOdometry(0, 2)

    estimated_path = []
    camera_pose_list = []

    start_translation = np.zeros((3, 1))
    start_rotation = np.identity(3)
    current_pose = np.concatenate((start_rotation, start_translation), axis=1)

    process_frames = False

    vo.read_images(show=True, gray=True)
    ret1, old_frame_left = vo.ret_left, vo.left_current_frame
    ret2, old_frame_right = vo.ret_right, vo.right_current_frame

    frame_counter: int = 0

    while True:

        vo.read_images(show=True, gray=True)
        ret1, new_frame_left = vo.ret_left, vo.left_current_frame
        ret2, new_frame_right = vo.ret_right, vo.right_current_frame

        frame_counter += 1
        start = time.perf_counter()

        if process_frames and ret1 and ret2:
            left_keypoints1 = vo.get_tiled_keypoints(old_frame_left, 10, 20)

            tp1_l, tp2_l = vo.track_keypoints(old_frame_left, new_frame_left, left_keypoints1)

            old_disp = np.divide(vo.disparity.compute(old_frame_left, old_frame_right).astype(np.float32), 16)
            new_disp = np.divide(vo.disparity.compute(new_frame_left, new_frame_right).astype(np.float32), 16)

            tp1_l, tp1_r, tp2_l, tp2_r = vo.calculate_right_qs(tp1_l, tp2_l, old_disp, new_disp)

            Q1, Q2 = vo.calc_3d(tp1_l, tp1_r, tp2_l, tp2_r)

            transformation_matrix = vo.estimate_pose(tp1_l, tp2_l, Q1, Q2)
            current_pose = current_pose @ transformation_matrix

        estimated_path.append((current_pose[0, 3], current_pose[2, 3]))

        old_frame_left = new_frame_left
        old_frame_right = new_frame_right
        process_frames = True
        end = time.perf_counter()
        total_time = end - start
        fps = 1 / total_time

        print(frame_counter)

        cv2.waitKey(1)

        if frame_counter == 50:
            break

    StereoVisualOdometry.visualize_path(estimated_path, 100)