import openpifpaf
from .parkingkp import ParkingKp



def register():
    openpifpaf.DATAMODULES['parkingkp'] = ParkingKp
