import serial


class Settings:

    PORT = "COM5"
    BAUDRATE = 921600
    DATA_LENGTH = 8
    STOP_BITS = serial.STOPBITS_ONE
    PARITY = serial.PARITY_NONE
    FLOW_CONTROL = False
