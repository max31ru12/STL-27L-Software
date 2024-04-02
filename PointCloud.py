class PointCloud:

    def __init__(self, array: list):
        # array of measurement-data
        self.data = array
        # LSB, MSB, intensity
        self.three_values = self.__distances()
        # PointCloud-data
        self.point_cloud = self.__make_parsed_dict()

    def __distances(self) -> list:
        result = []
        if len(self.data) % 3 == 0:
            for n in range(3, len(self.data) + 1, 3):
                result.append(tuple(self.data[n-3:n]))
            return result
        else:
            return []

    def __make_parsed_dict(self) -> dict:
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
