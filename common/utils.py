from typing import Any

from matplotlib.collections import PathCollection

from .models import PointCloud
import math
import matplotlib.pyplot as plt
import serial


def parse_data(data_sixteen: str) -> dict:
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


def from_pt_to_coordinates(point_cloud: dict) -> dict:
    coordinates = {}
    for key, point in point_cloud.items():
        x = math.sin(math.radians(point['angle'])) * point['distance']
        y = math.cos(math.radians(point['angle'])) * point['distance']
        coordinates[key] = {"x": round(x, 2), "y": round(y, 2)}

    return coordinates


def make_list_of_point_clouds(point_cloud: dict) -> tuple[list[Any], list[Any]]:
    x_coordinates = []
    y_coordinates = []
    try:
        for key, point in point_cloud.items():
            x = math.sin(math.radians(point['angle'])) * point['distance']
            y = math.cos(math.radians(point['angle'])) * point['distance']
            x_coordinates.append(round(x, 2))
            y_coordinates.append(round(y, 2))
    except KeyError:
        pass

    return x_coordinates, y_coordinates


def make_coordinates_from_pc_list(list_of_pc: list[dict]) -> (list, list):
    x_coord = []
    y_coord = []
    for _ in range(10):
        for point_cloud in list_of_pc:
            x_list, y_list = make_list_of_point_clouds(point_cloud)
            x_coord.extend(x_list)
            y_coord.extend(y_list)
    return x_coord, y_coord


def plot_points(x_coords: list, y_coords: list) -> PathCollection:
    """Creates a plot with provided coordinates"""
    try:
        plt.clf()
        # Построение точек
        scatter = plt.scatter(x_coords, y_coords, color='blue', marker='o')

        plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
        plt.axvline(0, color='black', linestyle='--', linewidth=0.5)

        # Добавление заголовка и меток осей
        plt.title('Облако точек')
        plt.xlabel('Ось X')
        plt.ylabel('Ось Y')
    except Exception as e:
        print(e)
        scatter = plt.scatter

    return scatter


def measure_one_spin(conn: serial.Serial) -> (list, list):
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
                        # return mapping
                        return make_coordinates_from_pc_list(mapping)

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
        return make_coordinates_from_pc_list(mapping)
        # return mapping
    except KeyboardInterrupt:
        conn.close()
