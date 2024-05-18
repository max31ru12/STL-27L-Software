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
    def __init__(self, straight_move: float, theta: float, prev: "MoveNode" = None) -> None:
        self.straight_move = straight_move  # прямое пермещение (в мм?)
        # угол поворота поворачивает текущую систему координат относительно предыдущего узла
        self.theta = radians(theta)  # угол поворота (в градусах?)
        self.prev = prev
        self.__total_theta = 0
        self.filter_set = set()

    def get_scanned_points(self):
        x, y = zip(*self.filter_set)
        return x, y

    def add_scanned_points(self, x_coords: list[float], y_coords: list[float]) -> None:
        self.filter_set.update(set(zip(x_coords, y_coords)))

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
        total_theta = 0
        current_node = self
        while current_node is not None:
            total_theta += current_node.theta
            current_node = current_node.prev
        return total_theta

    def calculate_bias(self) -> tuple[float, float]:
        """ Return current bias (for one move) relative to the start coordinate system """
        THETA = self.get_total_theta()
        x_bias = self.straight_move * cos(THETA)
        y_bias = self.straight_move * sin(THETA)
        return x_bias, y_bias

    def calculate_total_bias(self) -> tuple[float, float]:

        current_node = self
        x_total, y_total = 0, 0

        while current_node.prev is not None:

            x_total += current_node.calculate_bias()[0]
            y_total += current_node.calculate_bias()[1]
            current_node = current_node.prev

        x_total += current_node.calculate_bias()[0]
        y_total += current_node.calculate_bias()[1]

        return x_total, y_total

    def get_current_coordinates(self, x_coords: list[float], y_coords: list[float]):
        """
        Returns absolute coordinates matching the initial coordinate system
        (relative to the initial coordinate system)
        """
        x_bias, y_bias = self.calculate_total_bias()
        x_coords = [x + x_bias for x in x_coords]
        y_coords = [y + y_bias for y in y_coords]
        return self.rotate_points(x_coords, y_coords)


if __name__ == "__main__":

    n1 = MoveNode(straight_move=0, theta=0)
    bias1 = n1.calculate_bias()
    # print(degrees(n1.get_total_theta()))
    print(n1.calculate_bias())

    n2 = MoveNode(straight_move=1, theta=0, prev=n1)
    bias2 = n2.calculate_bias()
    # print(degrees(n2.get_total_theta()))
    print(n2.calculate_bias())

    n3 = MoveNode(straight_move=10, theta=90, prev=n2)
    bias3 = n3.calculate_bias()
    # print(degrees(n3.get_total_theta()))
    print(n3.calculate_bias())

    n4 = MoveNode(straight_move=2, theta=90, prev=n3)
    bias4 = n4.calculate_bias()
    # print(degrees(n4.get_total_theta()))
    print(n4.calculate_bias())
    #
    n5 = MoveNode(straight_move=5, theta=0, prev=n4)
    bias5 = n5.calculate_bias()
    # print(degrees(n5.get_total_theta()))
    print(n5.calculate_bias())

    total = (bias1[0] + bias2[0] + bias3[0] + bias4[0] + bias5[0], bias1[1] + bias2[1] + bias3[1] + bias4[1] + bias5[1])

    # print(total)

    print("\n\n\n\n\n\n")
    n5.calculate_total_bias()
