import time
from os import PathLike
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
        self.left_capture = cv2.VideoCapture(left_camera_number)
        self.right_capture = cv2.VideoCapture(right_camera_number)

        self.K_left, self.P_left = self.__load_calib("calib.txt")
        self.K_right, self.P_right = self.__load_calib("calib.txt")

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
        Loads the calibration of the camera
        Parameters
        ----------
        filepath (str): The file path to the camera file

        Returns
        -------
        K (ndarray): Intrinsic parameters
        P (ndarray): Projection matrix
        """
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]
        return K, P

    @property
    def current_frames_is_none(self) -> bool:
        return self.left_current_frame is None and self.right_current_frame

    @property
    def previous_frames_is_none(self) -> bool:
        return self.left_previous_frame is None and self.right_previous_frame

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

    @staticmethod
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def reprojection_residuals(self, dof, q1, q2, Q1, Q2):
        """
        Calculate the residuals

        Parameters
        ----------
        dof (ndarray): Transformation between the two frames. First 3 elements are the rotation vector and the last 3 is the translation. Shape (6)
        q1 (ndarray): Feature points in i-1'th image. Shape (n_points, 2)
        q2 (ndarray): Feature points in i'th image. Shape (n_points, 2)
        Q1 (ndarray): 3D points seen from the i-1'th image. Shape (n_points, 3)
        Q2 (ndarray): 3D points seen from the i'th image. Shape (n_points, 3)

        Returns
        -------
        residuals (ndarray): The residuals. In shape (2 * n_points * 2)
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
        Splits the image into tiles and detects the 10 best keypoints in each tile

        Parameters
        ----------
        img (ndarray): The image to find keypoints in. Shape (height, width)
        tile_h (int): The tile height
        tile_w (int): The tile width

        Returns
        -------
        kp_list (ndarray): A 1-D list of all keypoints. Shape (n_keypoints)
        """
        def get_kps(x, y):
            # Get the image tile
            impatch = img[y:y + tile_h, x:x + tile_w]

            # Detect keypoints
            keypoints = self.fast_features.detect(impatch)

            # Correct the coordinate for the point
            for pt in keypoints:
                pt.pt = (pt.pt[0] + x, pt.pt[1] + y)

            # Get the 10 best keypoints
            if len(keypoints) > 10:
                keypoints = sorted(keypoints, key=lambda x: -x.response)
                return keypoints[:10]
            return keypoints
        # Get the image height and width
        h, w, *_ = img.shape

        # Get the keypoints for each of the tiles
        kp_list = [get_kps(x, y) for y in range(0, h, tile_h) for x in range(0, w, tile_w)]

        # Flatten the keypoint list
        kp_list_flatten = np.concatenate(kp_list)
        return kp_list_flatten

    def track_keypoints(self, img1, img2, kp1, max_error: int = 4):

        trackpoints1 = np.expand_dims(cv2.KeyPoint_convert(kp1), axis=1)
        trackpoints2, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, trackpoints1, None, **self.lk_params)

        trackable = st.astype(bool)

        under_thresh = np.where(err[trackable] < max_error, True, False)

        trackpoints1 = trackpoints1[trackable][under_thresh]
        trackpoints2 = np.around(trackpoints2[trackable][under_thresh])

        h, w = img1.shape
        in_bounds = np.where(np.logical_and(trackpoints2[:, 1] < h, trackpoints2[:, 0] < w), True, False)

        trackpoints1 = trackpoints1[in_bounds]
        trackpoints2 = trackpoints2[in_bounds]

        return trackpoints1, trackpoints2

    def calculate_right_qs(self, q1, q2, disp1, disp2, min_disp=0.0, max_disp=100.0):

        def get_idxs(q, disp):
            q_idx = q.astype(int)
            disp = disp.T[q_idx[:, 0], q_idx[:, 1]]
            return disp, np.where(np.logical_and(min_disp < disp, disp < max_disp), True, False)

        disp1, mask1 = get_idxs(q1, disp1)
        disp2, mask2 = get_idxs(q2, disp2)

        in_bounds = np.logical_and(mask1, mask2)

        q1_l, q2_l, disp1, disp2 = q1[in_bounds], q2[in_bounds], disp1[in_bounds], disp2[in_bounds]

        q1_r, q2_r = np.copy(q1_l), np.copy(q2_l)

        q1_r[:, 0] -= disp1
        q2_r[:, 0] -= disp2

        return q1_l, q1_r, q2_l, q2_r

    def calc_3d(self, q1_l, q1_r, q2_l, q2_r):
        # triangulate points from i-1'th image
        Q1 = cv2.triangulatePoints(self.P_left, self.P_right, q1_l.T, q1_r.T)
        Q1 = np.transpose(Q1[:3] / Q1[3])

        # triangulate points from i'th image
        Q2 = cv2.triangulatePoints(self.P_left, self.P_right, q2_l.T, q2_r.T)
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
            transformation_matrix = self._form_transf(R, t)

            return transformation_matrix

    def get_pose(self, old_imgL, old_imgR, new_imgL, new_imgR):

        img1_l = old_imgL
        img2_l = new_imgL

        kp1_l = self.get_tiled_keypoints(img1_l, 10, 20)

        tp1_l, tp2_l = self.track_keypoints(img1_l, img2_l, kp1_l)

        old_disp = np.divide(self.disparity.compute(old_imgL, old_imgR).astype(np.float32), 16)
        new_disp = np.divide(self.disparity.compute(new_imgL, new_imgR).astype(np.float32), 16)

        tp1_l, tp1_r, tp2_l, tp2_r = self.calculate_right_qs(tp1_l, tp2_l, old_disp, new_disp)

        # Условие, проверяющее отсутствие совпадений
        try:
            if tp1_l.size == 0 or tp2_l.size == 0 or tp1_r.size == 0 or tp2_r.size == 0:
                print("Не удалось обнаружить совпадающие точки. Пропуск текущего шага.")
                Q1, Q2 = self.calc_3d(*self.previous_matches)
            else:
                Q1, Q2 = self.calc_3d(tp1_l, tp1_r, tp2_l, tp2_r)
        except Exception as e:
            print(tp1_l, tp1_r, tp2_l, tp2_r, sep="\n\n\n")
            raise e

        self.previous_matches = [tp1_l, tp1_r, tp2_l, tp2_r]
        transformation_matrix = self.estimate_pose(tp1_l, tp2_l, Q1, Q2)

        return transformation_matrix


if __name__ == "__main__":
    skip_frames = 2
    vo = StereoVisualOdometry(2, 0)

    gt_path = []
    estimated_path = []
    camera_pose_list = []

    start_translation = np.zeros((3, 1))
    start_rotation = np.identity(3)
    start_pose = np.concatenate((start_rotation, start_translation), axis=1)

    process_frames = False
    vo.read_images(show=True, gray=True)
    ret1, old_frame_left = vo.ret_left, vo.left_current_frame
    ret2, old_frame_right = vo.ret_right, vo.right_current_frame
    new_frame_left = None
    new_frame_right = None

    cur_pose = start_pose
    frame_counter: int = 0

    while True:
        # Чтение новых кадров
        vo.read_images(show=True, gray=True)
        ret1, new_frame_left = vo.ret_left, vo.left_current_frame
        ret2, new_frame_right = vo.ret_right, vo.right_current_frame

        frame_counter += 1

        start = time.perf_counter()

        # Если процесс кадров активен и новые кадры успешно считаны
        if process_frames and ret1 and ret2:
            transf = vo.get_pose(old_frame_left, old_frame_right, new_frame_left, new_frame_right)
            cur_pose = cur_pose @ transf
            hom_array = np.array([[0, 0, 0, 1]])
            hom_camera_pose = np.concatenate((cur_pose, hom_array), axis=0)
            camera_pose_list.append(hom_camera_pose)
            estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))

        # Обновление предыдущих кадров текущими
        old_frame_left = new_frame_left
        old_frame_right = new_frame_right
        process_frames = True
        end = time.perf_counter()
        total_time = end - start
        fps = 1 / total_time

        cv2.putText(new_frame_left, str(np.round(cur_pose[0, 0], 2)), (260, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (255, 0, 0), 1)
        cv2.putText(new_frame_left, str(np.round(cur_pose[0, 0], 2)), (260, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (255, 0, 0), 1)
        cv2.putText(new_frame_left, str(np.round(cur_pose[0, 1], 2)), (260, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (255, 0, 0), 1)
        cv2.putText(new_frame_left, str(np.round(cur_pose[0, 2], 2)), (260, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (255, 0, 0), 1)
        cv2.putText(new_frame_left, str(np.round(cur_pose[1, 0], 2)), (260, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (255, 0, 0), 1)
        cv2.putText(new_frame_left, str(np.round(cur_pose[1, 1], 2)), (260, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (255, 0, 0), 1)
        cv2.putText(new_frame_left, str(np.round(cur_pose[1, 2], 2)), (260, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (255, 0, 0), 1)
        cv2.putText(new_frame_left, str(np.round(cur_pose[2, 0], 2)), (260, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (255, 0, 0), 1)
        cv2.putText(new_frame_left, str(np.round(cur_pose[2, 1], 2)), (260, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (255, 0, 0), 1)
        cv2.putText(new_frame_left, str(np.round(cur_pose[2, 2], 2)), (260, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (255, 0, 0), 1)
        cv2.putText(new_frame_left, str(np.round(cur_pose[0, 3], 2)), (540, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (255, 0, 0), 1)
        cv2.putText(new_frame_left, str(np.round(cur_pose[1, 3], 2)), (540, 90), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (255, 0, 0), 1)
        cv2.putText(new_frame_left, str(np.round(cur_pose[2, 3], 2)), (540, 130), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (255, 0, 0), 1)

        cv2.imshow("img", new_frame_left)
        cv2.imshow("img2", new_frame_right)

        cv2.waitKey(1)

        print(frame_counter)
        if frame_counter == 50:
            break

    StereoVisualOdometry.visualize_path(estimated_path, 100)
