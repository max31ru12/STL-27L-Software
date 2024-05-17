import time
from serial import Serial

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from common.models import MoveNode
from common.utils import plot_points, measure_one_spin
from common.config import Settings
from common.filters import filter_noises

SERIAL = Serial(Settings.PORT,
                baudrate=Settings.BAUDRATE,
                stopbits=Settings.STOP_BITS,
                bytesize=Settings.DATA_LENGTH,
                parity=Settings.PARITY,
                xonxoff=Settings.FLOW_CONTROL)


def update(frame):
    x_coord, y_coord = measure_one_spin(SERIAL)
    x_coord, y_coord = filter_noises(x_coord, y_coord)
    # print(max(x_coord), min(x_coord))
    # print(max(y_coord), min(y_coord))
    # print("\n\n\n")

    # print(sum(x_coord)/len(x_coord))
    scatter = plot_points(x_coord, y_coord)

    # Здесь так почему-то надо
    return scatter,


# animation = FuncAnimation(fig, update, interval=2000, blit=True, cache_frame_data=False)

# plt.show()

filter_set = set()
current_node = MoveNode(straight_move=0, theta=0)

for i in range(1, 20):

    print(i)
    x_coord, y_coord = measure_one_spin(SERIAL)

    x_coord, y_coord = filter_noises(x_coord, y_coord)

    x_coord, y_coord = current_node.rotate_points(x_coord, y_coord)
    # x_coord, y_coord = current_node.get_current_coordinates(x_coord, y_coord)

    filter_set.update(set(zip(x_coord, y_coord)))
    x, y = zip(*filter_set)

    if i % 10 == 0:
        move = float(input("Введите перемещение: "))
        theta = float(input("Введите угол: "))
        current_node = MoveNode(straight_move=move, theta=theta, prev=current_node)
        SERIAL.flushInput()


plt.scatter(x, y, s=0.1)
plt.show()