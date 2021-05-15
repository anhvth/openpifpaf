import openpifpaf
from .parking_line_kp import ParkingLineKp



def register():
    openpifpaf.DATAMODULES['parking_line_kp'] = ParkingLineKp
