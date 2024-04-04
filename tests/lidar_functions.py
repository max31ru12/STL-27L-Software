import time

import numpy as np
import serial
from pprint import pprint

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from utils import parse_data, interpolation, Settings, transfer_to_coordinates, plot_points

SERIAL = serial.Serial(Settings.PORT,
                       baudrate=Settings.BAUDRATE,
                       stopbits=Settings.STOP_BITS,
                       bytesize=Settings.DATA_LENGTH,
                       parity=Settings.PARITY,
                       xonxoff=Settings.FLOW_CONTROL)


def measure_one_spin(conn: serial.Serial) -> list[dict]:
    """Returns a list of dicts of PointCloud's per 360 degrees"""
    result = {}
    counter = 0  # переменная для работы с получаемыми данными как со списком
    prev = "prev"
    prev_start_angle: int = 0
    mapping = []

    try:
        while True:
            data = conn.read()
            hex_data = ' '.join([hex(byte)[2:].zfill(2) for byte in data])
            result[0] = ""

            if prev == "54" and hex_data == "2c":
                out_data = parse_data(result[counter])
                if "error" not in out_data.keys():

                    point_cloud = out_data["PointCloud"].point_cloud
                    start_angle = out_data["start_angle"]
                    end_angle = out_data["end_angle"]
                    angle_difference = prev_start_angle - start_angle

                    interpolation(point_cloud, start_angle, end_angle)

                    if 356 < angle_difference < 360:
                        # print(f"{start_angle=}")
                        # print(f"{prev_start_angle=}")
                        # print(f"{angle_difference=}")
                        # print("\n")
                        return mapping

                    prev_start_angle = start_angle

                    if "error" not in point_cloud:
                        mapping.append(point_cloud)

                counter += 1
                result[counter] = ""
                result[counter] += ("54 " + "2c ")
            else:
                temp = False
                if hex_data == "54":
                    prev = hex_data
                    continue
                if temp and hex_data != "2c":
                    result[counter] += f"54 "
                result[counter] += f"{hex_data} "
            prev = hex_data
        return mapping
    except KeyboardInterrupt:
        conn.close()


def make_coordinates_from_pc(list_of_pc: list[dict]) -> tuple[list, list]:
    x_coord = []
    y_coord = []
    for _ in range(10):
        for point_cloud in list_of_pc:
            x_list, y_list = transfer_to_coordinates(point_cloud)
            x_coord.extend(x_list)
            y_coord.extend(y_list)
    # x_coord = [item for sublist in x_coord for item in sublist]
    # y_coord = [item for sublist in y_coord for item in sublist]
    return x_coord, y_coord


# Create a figure and axis
fig, ax = plt.subplots()
sc = ax.scatter([], [])

# ax.set_xlim(-1000, 1000)  # Change limits according to your data range
# ax.set_ylim(-1000, 1000)  # Change limits according to your data range


def update(frame):
    data = measure_one_spin(SERIAL)
    x_coord, y_coord = make_coordinates_from_pc(data)

    try:
        plt.clf()

        scatter = plt.scatter(x_coord, y_coord, color='blue', marker='o')

        plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
        plt.axvline(0, color='black', linestyle='--', linewidth=0.5)

        plt.title('Облако точек')
        plt.xlabel('Ось X')
        plt.ylabel('Ось Y')

    except Exception as e:

        print(e)

        plt.clf()
        scatter = plt.scatter(0, 0, color='blue', marker='o')

    return scatter,


animation = FuncAnimation(fig, update, interval=2000, blit=True, cache_frame_data=False)

plt.show()


# for i in range(10):
#     data = measure_one_spin(SERIAL)
#     coordinates_tuple = make_coordinates_from_pc(data)
#
#     plt = plot_points(*coordinates_tuple)
#     time.sleep(0.5)
#
#     plt.close()




