import cv2
import numpy as np
import matplotlib.pyplot as plt

from visual_odometry import VisualOdometry


plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
line, = ax.plot([], [], marker='o')
ax.set_xlim(-100, 100)  # Adjust based on expected path range
ax.set_ylim(-100, 100)
ax.set_xlabel('X axis')
ax.set_ylabel('Z axis')
ax.set_title('Real-Time Estimated Path')


odometry = VisualOdometry(0, 0.95)
current_pose = np.eye(4)
estimated_path = []

i = 0
while True:

    odometry.read_image(gray=False)

    prev_image_matches, cur_image_matches = odometry.get_matches()
    if prev_image_matches is None and cur_image_matches is None:
        continue
    transf = odometry.get_pose(prev_image_matches, cur_image_matches)
    if np.isnan(transf).any() or np.isinf(transf).any():
        print("Ошибка: матрица трансформации содержит NaN или бесконечные значения")
        continue
    current_pose = np.matmul(current_pose, np.linalg.inv(transf))

    estimated_path.append((current_pose[0, 3], current_pose[2, 3]))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    x, y = zip(*estimated_path)
    line.set_data(x, y)
    ax.relim()
    ax.autoscale_view(True, True, True)
    plt.draw()
    plt.pause(0.01)

x, y = zip(estimated_path)

plt.plot(x, y, marker='o')