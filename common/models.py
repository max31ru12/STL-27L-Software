from math import sin, cos, radians, degrees


class PointCloud:

    def __init__(self, lidar_data: list):
        self.data = lidar_data  # array of gotten measured data
        self.three_values = self.__distances()  # LSB, MSB, intensity

    def __distances(self) -> list:
        result = []
        if len(self.data) % 3 == 0:
            for n in range(3, len(self.data) + 1, 3):
                result.append(tuple(self.data[n - 3:n]))
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
                point_cloud[key] = {
                    "distance": value,
                    "intensity": intensity
                }
                count += 1
            return point_cloud
        else:
            point_cloud = {"error": "incorrect length of data"}
            return point_cloud

    def __str__(self):
        return "PointCloud-object"


class MoveNode:
    """Linked-List-like data structure"""

    counter = 0

    def __new__(cls, *args, **kwargs):
        cls.counter += 1
        return super().__new__(cls)

    def __init__(self, straight_move: float, theta: float, prev: "MoveNode" = None) -> None:
        self.straight_move = straight_move  # прямое пермещение (в мм?)
        # угол поворота поворачивает текущую систему координат относительно предыдущего узла
        self.theta = radians(theta)  # угол поворота (в градусах?)
        self.prev = prev
        self.ppp = self.counter
        self.__total_theta = 0

    def __str__(self):
        return f"NODE: {self.ppp}"

    def rotate_points(self, x_coords: list[float], y_coords: list[float]) -> (list[float], list[float]):
        """ метод поворачивает систему координат """
        rotated_x = []
        rotated_y = []
        for x, y in zip(x_coords, y_coords):
            new_x = x * cos(self.theta) - y * sin(self.theta)
            new_y = x * sin(self.theta) + y * cos(self.theta)
            rotated_x.append(new_x)
            rotated_y.append(new_y)
        return rotated_x, rotated_y

    def get_total_theta(self) -> float:
        """ Return total angle in radians """
        current_node = self
        self.__total_theta += self.theta
        while current_node.prev is not None:
            self.__total_theta += current_node.prev.theta
            current_node = current_node.prev
        return self.__total_theta

    def calculate_node_bias(self) -> tuple[float, float]:
        THETA = self.get_total_theta()
        x_bias = self.straight_move * cos(THETA)
        y_bias = self.straight_move * sin(THETA)
        return x_bias, y_bias


n1 = MoveNode(straight_move=0, theta=0)

n2 = MoveNode(straight_move=1, theta=0, prev=n1)
print(n2.calculate_node_bias())

n3 = MoveNode(straight_move=10, theta=90, prev=n2)
print(n3.calculate_node_bias())

# n4 = MoveNode(straight_move=2, theta=90, prev=n3)
# print(n4.calculate_node_bias())
#
# n5 = MoveNode(straight_move=5, theta=0, prev=n4)
# print(n5.calculate_node_bias())
