from math import sin, cos, radians, degrees
from matplotlib import pyplot as plt


class PointCloud:

    def __init__(self, lidar_data: list):
        # array of gotten measured data
        self.data = lidar_data
        # LSB, MSB, intensity
        self.three_values = self.__distances()

    def __distances(self) -> list:
        result = []
        if len(self.data) % 3 == 0:
            for n in range(3, len(self.data) + 1, 3):
                result.append(tuple(self.data[n-3:n]))
            return result
        else:
            return []

    @property
    def point_cloud(self) -> dict:
        point_cloud = {}
        count = 1
        if len(self.three_values):
            for point in self.three_values:
                key = f"Point {count}"
                value = int(point[1].strip() + point[0].strip(), 16)
                intensity = int(point[2], 16)
                point_cloud[key] = {"distance": value,
                                    "intensity": intensity}
                count += 1
            return point_cloud
        else:
            point_cloud = {"error": "incorrect length of data"}
            return point_cloud

    def __str__(self):
        return "PointCloud-object"


class MoveNode:
    # уточнить единицы измерения
    def __init__(self, straight_move: float, theta: float, prev: "MoveNode" = None) -> None:
        self.straight_move = straight_move  # прямое пермещение (в мм?)
        self.theta = radians(theta)  # угол поворота (в градусах?)
        self.prev = prev

    @property
    def coordinate_bias(self) -> tuple[float, float]:
        x_d = self.straight_move * cos(self.theta)
        y_d = self.straight_move * sin(self.theta)
        return x_d, y_d

    @property
    def path(self) -> tuple[float, float, float]:
        x_path, y_path = self.coordinate_bias
        current_theta = self.theta
        current_node = self
        while current_node.prev is not None:
            x_bias, y_bias = self.prev.coordinate_bias
            x_path += x_bias
            y_path += y_bias
            current_theta += self.prev.theta
            current_node = current_node.prev

        return x_path, y_path, degrees(current_theta)

    def delete_bias(self, x_coords: list, y_coords: list) -> tuple[list, list]:
        x_bias, y_bias, theta_bias = self.path
        x_coords = [x - x_bias for x in x_coords]
        y_coords = [y - y_bias for y in y_coords]
        return x_coords, y_coords


x, y = [1, 2, 3], [1, 2, 3]

x_2, y_2 = [409, 410, 411], [1, 2, 3]

plt.scatter(x, y)
plt.show()

node_1 = MoveNode(straight_move=4, theta=0)
print(node_1.prev)
node_2 = MoveNode(straight_move=4, theta=0, prev=node_1)
print(node_2.prev)
node_3 = MoveNode(straight_move=400, theta=0, prev=node_2)
print(node_3.prev)

print(node_3.delete_bias(x_2, y_2))

