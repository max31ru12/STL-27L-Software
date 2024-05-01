import time

import serial


from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from common.utils import plot_points, measure_one_spin
from common.config import Settings
from common.filters import filter_noises


SERIAL = serial.Serial(Settings.PORT,
                       baudrate=Settings.BAUDRATE,
                       stopbits=Settings.STOP_BITS,
                       bytesize=Settings.DATA_LENGTH,
                       parity=Settings.PARITY,
                       xonxoff=Settings.FLOW_CONTROL)
# fig, ax = plt.subplots()


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

for i in range(100):
    x_coord, y_coord = measure_one_spin(SERIAL)
    x_coord, y_coord = filter_noises(x_coord, y_coord)



    filter_set.update(set(zip(x_coord, y_coord)))
    x, y = zip(*filter_set)


    print(len(x))

    # filter_set.update()

    plt.scatter(x, y, s=0.1)
    plt.show()

    # plt = plot_points(x_coord, y_coord)
    time.sleep(0.5)
    plt.close()



