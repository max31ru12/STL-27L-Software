import serial
from pprint import pprint
from utils import parse_data, interpolation, from_pt_to_coordinates, plot_points, Settings

SERIAL = serial.Serial(Settings.PORT,
                       baudrate=Settings.BAUDRATE,
                       stopbits=Settings.STOP_BITS,
                       bytesize=Settings.DATA_LENGTH,
                       parity=Settings.PARITY,
                       xonxoff=Settings.FLOW_CONTROL)


def get_lidar_data(conn: serial.Serial):

    result = {}
    graphic_data = []
    key = 0
    prev = "prev"
    count = 0

    try:
        while True:
            count += 1
            data = conn.read()
            hex_data = ' '.join([hex(byte)[2:].zfill(2) for byte in data])
            result[0] = ""
            if prev == "54" and hex_data == "2c":
                # Из out_data можно получить все данные
                out_data = parse_data(result[key])

                # # Блок выводит данные облака точек, если они не ошибка
                if "error" not in out_data.keys():

                    point_cloud = out_data["PointCloud"].point_cloud
                    start_angle = out_data["start_angle"]
                    end_angle = out_data["end_angle"]

                    print(f"{start_angle=}, {end_angle=}")

                    # дополняет PointCloud.point_cloud ключом angle
                    interpolation(point_cloud, start_angle, end_angle)

                    # После интерполяции добавляем данные в список данных для графики
                    # Чтобы не попадали ошибки
                    if "error" not in point_cloud:
                        graphic_data.append(from_pt_to_coordinates(point_cloud))
                key += 1
                result[key] = ""
                result[key] += ("54 " + "2c ")
            else:
                # Блок нужен для того, чтобы отсечь лишнюю 54 в конце
                temp = False
                if hex_data == "54":
                    temp = True
                    prev = hex_data
                    continue
                if temp and hex_data != "2c":
                    result[key] += f"54 "
                result[key] += f"{hex_data} "
            prev = hex_data

    except KeyboardInterrupt:
        conn.close()

    return graphic_data


data = get_lidar_data(SERIAL)
# print(data)


