import matplotlib.pyplot as plt
import math
from pprint import pprint

# Ваши данные
data = {
        'Point 1': {'angle': 251.48, 'distance': 4638, 'intensity': 70},
        'Point 10': {'angle': 252.97727272727272, 'distance': 4508, 'intensity': 70},
        'Point 11': {'angle': 253.14363636363638, 'distance': 4492, 'intensity': 70},
        'Point 12': {'angle': 253.31, 'distance': 4482, 'intensity': 71},
        'Point 2': {'angle': 251.64636363636362, 'distance': 4623, 'intensity': 70},
        'Point 3': {'angle': 251.81272727272727, 'distance': 4607, 'intensity': 70},
        'Point 4': {'angle': 251.9790909090909, 'distance': 4589, 'intensity': 70},
        'Point 5': {'angle': 252.14545454545453, 'distance': 4575, 'intensity': 71},
        'Point 6': {'angle': 252.31181818181818, 'distance': 4561, 'intensity': 70},
        'Point 7': {'angle': 252.4781818181818, 'distance': 4548, 'intensity': 70},
        'Point 8': {'angle': 252.64454545454547, 'distance': 4534, 'intensity': 70},
        'Point 9': {'angle': 252.8109090909091, 'distance': 4521, 'intensity': 70}
        }


def from_pt_to_coordinates(point_cloud: dict):

    coordinates = {}

    for key, point in point_cloud.items():
        x = math.sin(math.radians(point['angle'])) * point['distance']
        y = math.cos(math.radians(point['angle'])) * point['distance']
        coordinates[key] = {"x": round(x, 2), "y": round(y, 2)}

    return coordinates


x_coord = []
y_coord = []

new_data = from_pt_to_coordinates(data)
for key in new_data.keys():
    x_coord.append(new_data[key]["x"])
    y_coord.append(new_data[key]["y"])


# pprint(from_pt_to_coordinates(data))




def plot_points(x_coords, y_coords):
    # Построение точек
    plt.scatter(x_coords, y_coords, color='blue', marker='o')

    # Добавление красной точки в (0, 0)
    zero_point_color = 'red'
    zero_point_size = 100  # Размер красной точки

    # Проверка каждой точки
    for x, y in zip(x_coords, y_coords):
        if x == 0 and y == 0:
            plt.scatter(x, y, color=zero_point_color, marker='o', s=zero_point_size)

    # Добавление заголовка и меток осей
    plt.title('График точек')
    plt.xlabel('Ось X')
    plt.ylabel('Ось Y')

    # Отображение графика
    plt.show()


plot_points(x_coord, y_coord)







# # Инициализация графика
# fig, ax = plt.subplots()
# points, = ax.plot([], [], 'bo')  # Используем круглые маркеры для точек
#
# # Функция инициализации анимации
# def init():
#     ax.set_xlim(-100000, 100000)  # Задайте пределы для x и y
#     ax.set_ylim(-300000, 300000)
#     return points,
#
# # Функция обновления графика
# def update(frame):
#     x = [data[point]["angle"] for point in data]
#     y = [data[point]["distance"] for point in data]
#     points.set_data(x, y)
#     return points,
#
# # Создание анимации
# ani = FuncAnimation(fig, update, frames=None, init_func=init, blit=True)
#
# plt.show()