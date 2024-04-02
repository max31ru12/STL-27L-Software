from PointCloud import PointCloud
import math
import matplotlib.pyplot as plt
import serial


class Settings:

    PORT = "COM5"
    BAUDRATE = 921600
    DATA_LENGTH = 8
    STOP_BITS = serial.STOPBITS_ONE
    PARITY = serial.PARITY_NONE
    FLOW_CONTROL = False


def parse_data(data_sixteen: str):
    DATA = {}

    if len(data_sixteen) != 0:
        data_list = data_sixteen.split()
        N = len(data_list)
        try:

            DATA["header"] = data_list[0]
            DATA["VevLen"] = data_list[1]
            DATA["speed"] = int(data_list[3] + data_list[2], 16)
            DATA["start_angle"] = int(data_list[5] + data_list[4], 16) / 100
            DATA["PointCloud"] = PointCloud(data_list[6:N - 5])
            DATA["end_angle"] = int(data_list[N - 4] + data_list[N - 5], 16) / 100
            DATA["timestamp"] = int(data_list[N - 2] + data_list[N - 3], 16)
            DATA["CRC_check"] = int(data_list[N - 1], 16)

        except Exception as E:
            print(E, "data is less than 11 bytes")
            DATA["error"] = "incorrect data"

    else:
        DATA["error"] = "incorrect data"

    return DATA


def interpolation(point_cloud: dict, start_angle: float, end_angle: float) -> dict:
    """Функция интерполяции точек по углу: дополняет point_cloud полем angle"""
    if 'error' not in point_cloud.keys():
        if len(point_cloud) > 2: # Это костыль нужен тогда, когда в измерении всего одна точка
            step = (end_angle - start_angle) / (len(point_cloud) - 1)

            for i in range(0, len(point_cloud)):
                key = f"Point {i + 1}"
                angle = start_angle + step * i
                point_cloud[key]["angle"] = angle
        else: # Это костыль нужен тогда, когда в измерении всего одна точка
            point_cloud["Point 1"]["angle"] = (end_angle - start_angle) / 2
    else:
        point_cloud["interpolation"] = "Interpolation Error"
    return point_cloud


def from_pt_to_coordinates(point_cloud: dict):
    coordinates = {}
    for key, point in point_cloud.items():
        x = math.sin(math.radians(point['angle'])) * point['distance']
        y = math.cos(math.radians(point['angle'])) * point['distance']
        coordinates[key] = {"x": round(x, 2), "y": round(y, 2)}

    return coordinates


def plot_points(x_coords, y_coords):
    # Построение точек
    plt.scatter(x_coords, y_coords, color='blue', marker='o')

    plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.5)

    # Добавление заголовка и меток осей
    plt.title('Облако точек')
    plt.xlabel('Ось X')
    plt.ylabel('Ось Y')

    # Отображение графика
    plt.show()

