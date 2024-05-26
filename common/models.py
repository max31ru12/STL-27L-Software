from math import sin, cos, radians  # degrees


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


def compute_next_pose(current_pose: [float, float, float], control: [float, float, float]):
    x, y, theta = current_pose
    delta_x, delta_y, delta_theta = control
    return [x + delta_x, y + delta_y, theta + delta_theta]


class Map:

    pass


class Vertex:

    def __init__(self, pose: list[float, float, float]) -> None:
        self.pose = pose  # [x, y, theta]
        self.point_cloud: list[list[float], list[float]] | None = None

    def rotate_point_cloud(self) -> (list[float], list[float]):
        """ метод поворачивает систему координат облака точек """
        theta = self.pose[2]
        x_coords, y_coords = self.point_cloud[0], self.point_cloud[1]
        rotated_x = []
        rotated_y = []
        for x, y in zip(x_coords, y_coords):
            new_x = x * cos(theta) - y * sin(theta)
            new_y = x * sin(theta) + y * cos(theta)
            rotated_x.append(new_x)
            rotated_y.append(new_y)
        return rotated_x, rotated_y


class Edge:

    # Храним показания датчиков, узлы i и j
    def __init__(self, start: Vertex, end: Vertex, measurement: list[float, float, float]) -> None:
        self.start_vertex = start  # i
        self.end_vertex = end  # j
        self.measurement = measurement  # Z (dx, dy, dtheta) - это показания одометра
        self.prediction = self.__calculate_prediction()  # Z с крышкой (dx, dy, dtheta)

    def __calculate_prediction(self) -> list[float, float, float]:
        """ Предсказываем на основе данных одометрии и данных предыдущего узла """
        x_i, y_i, theta_i = self.start_vertex.pose
        dx, dy, dtheta = self.measurement

        # Предсказанное относительное положение (с учетом текущей ориентации theta_i)
        pred_x = x_i + dx * cos(theta_i) - dy * sin(theta_i)
        pred_y = y_i + dx * sin(theta_i) + dy * cos(theta_i)
        pred_theta = theta_i + dtheta
        return [pred_x, pred_y, pred_theta]

    def compute_error(self) -> list[float, float, float]:
        """ Считаем разницу между измеренными данными и предсказанными (разница между векторами состояния) """
        z_pose_j = self.end_vertex.pose  # Измеренное относительное положение вершины j
        z_prediction = self.prediction  # Предсказанное относительное положение вершины j
        e_ij = [z_pose_j[i] - z_prediction[i] for i in range(len(z_pose_j))]  # Вектор ошибки
        return e_ij


if __name__ == "__main__":

    # старт
    v1 = Vertex([263, 77, 0])
    print(f"{v1.pose=}")

    # измерение одометрии (x, y, theta)
    odometry = (1000, 500, 30)
    next_pose = compute_next_pose(v1.pose, odometry)

    # Вторая вершина
    v2 = Vertex(next_pose)
    print(v2.pose)

    odometry = (400, 50000, 147)
    next_pose = compute_next_pose(v2.pose, odometry)
    v3 = Vertex(next_pose)
    print(v3.pose)

    # e12 = Edge(v1, v2, odometry)
