import time
from math import degrees

from serial import Serial

from matplotlib import pyplot as plt

from common.models import MoveNode
from common.utils import measure_one_spin, plot_lidar_localization, plot_lidar_lines
from common.config import Settings
from common.filters import filter_noises

SERIAL = Serial(Settings.PORT,
                baudrate=Settings.BAUDRATE,
                stopbits=Settings.STOP_BITS,
                bytesize=Settings.DATA_LENGTH,
                parity=Settings.PARITY,
                xonxoff=Settings.FLOW_CONTROL)

filter_set = set()
current_node = MoveNode(straight_move=0, theta=0)
plt.scatter(x=0, y=0, color='red')


for i in range(1, 20):

    print(i)

    x_coords, y_coords = measure_one_spin(SERIAL)
    x_coords, y_coords = filter_noises(x_coords, y_coords)
    x_coords, y_coords = current_node.get_current_coordinates(x_coords, y_coords)

    filter_set.update(set(zip(x_coords, y_coords)))
    x, y = zip(*filter_set)

    current_node.add_scanned_points(x_coords, y_coords)

    if i % 10 == 0:
        move = float(input("Введите перемещение: "))
        theta = float(input("Введите угол: "))
        current_node = MoveNode(straight_move=move, theta=theta, prev=current_node)
        plot_lidar_localization(current_node)
        SERIAL.flushInput()


plt.scatter(x, y, s=0.1)

plt.show()
