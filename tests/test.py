import time

import cv2
import numpy as np
import matplotlib.pyplot as plt

from visual_odometry_refactor import VisualOdometry

plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
line, = ax.plot([], [], marker='o')
ax.set_xlim(-100, 100)  # Adjust based on expected path range
ax.set_ylim(-100, 100)
ax.set_xlabel('X axis')
ax.set_ylabel('Z axis')
ax.set_title('Real-Time Estimated Path')

odometry = VisualOdometry(1, 0.75)
current_pose = np.eye(4)
estimated_path = []

i = 0
while True:

    odometry.read_image(gray=True, show=False)
    odometry.get_matches(show=True)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
