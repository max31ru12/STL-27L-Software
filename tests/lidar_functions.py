import serial
from pprint import pprint
from utils import parse_data, interpolation, from_pt_to_coordinates, plot_points, Settings, transfer_to_coordinates
import time


SERIAL = serial.Serial(Settings.PORT,
                       baudrate=Settings.BAUDRATE,
                       stopbits=Settings.STOP_BITS,
                       bytesize=Settings.DATA_LENGTH,
                       parity=Settings.PARITY,
                       xonxoff=Settings.FLOW_CONTROL)


def measure_one_spin(conn: serial.Serial) -> list[dict]:
    """Returns a list of dicts of PointCloud's """

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
                        print(f"{start_angle=}")
                        print(f"{prev_start_angle=}")
                        print(f"{angle_difference=}")
                        print("\n")
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


x_coord = []
y_coord = []
for i in range(10):
    data = measure_one_spin(SERIAL)
    for point_cloud in data:
        x_list, y_list = transfer_to_coordinates(point_cloud)
        x_coord.append(x_list)
        y_coord.append(y_list)
    plt = plot_points(x_coord, y_coord)
    time.sleep(2)
