import numpy as np

from PIL import Image

class Stats():
    def __init__(self, app_ref):
        self.app_ref = app_ref

    def proc_pixel_value(self, x, y):
        I = np.array(self.app_ref.img_proc)
        color = I[y][x]

        if type(color) == np.bool_:
            return  1 if color else 0
        return color
