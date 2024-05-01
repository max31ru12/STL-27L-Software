import math


def filter_noises(x_coordinates: list, y_coordinates: list) -> (list, list):
    x_coordinates_filtered = []
    y_coordinates_filtered = []

    for i in range(len(x_coordinates)):
        radius = math.sqrt(x_coordinates[i] ** 2 + y_coordinates[i] ** 2)
        if radius < 10000:
            x_coordinates_filtered.append(x_coordinates[i])
            y_coordinates_filtered.append(y_coordinates[i])

    return x_coordinates_filtered, y_coordinates_filtered

