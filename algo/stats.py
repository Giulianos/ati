import numpy as np

from PIL import Image

import tkinter as tk

class Stats():
    def __init__(self, app_ref):
        self.app_ref = app_ref

    def proc_pixel_value(self, x, y):
        I = np.array(self.app_ref.img_proc)
        color = I[y][x]

        if type(color) == np.bool_:
            return  1 if color else 0
        return color

    def region_mean_value(self):
        # Request region selection
        self.app_ref.mouse_selection.request_selection(self.handle_region_mean_value)

    def handle_region_mean_value(self, start, end):
        img = self.app_ref.img_proc
        bands = self.app_ref.img_proc.getbands()
        I = np.array(img)
        x_start, y_start = start
        x_end, y_end = end
        if bands == ('1',): # binary
            avg = self.mean_binary(I, x_start, y_start, x_end, y_end)
        elif bands == ('L',): # grayscale
            avg = self.mean_grayscale(I, x_start, y_start, x_end, y_end)
        elif bands == ('R','G','B'): # rgb
            avg = self.mean_rgb(I, x_start, y_start, x_end, y_end)

        tk.messagebox.showinfo('Promedio', 'El promedio de los pixeles en la region seleccionada es {}'.format(avg))

    def mean_rgb(self, I, x1, y1, x2, y2):
        pixel_count = 0
        avg = (0,0,0)
        for x in range(x1, x2+1):
            for y in range(y1, y2+1):
                avg += I[y][x]
                pixel_count += 1

        avg[0] /= pixel_count
        avg[1] /= pixel_count
        avg[2] /= pixel_count

        return avg

    def mean_grayscale(self, I, x1, y1, x2, y2):
        pixel_count = 0
        avg = 0
        for x in range(x1, x2+1):
            for y in range(y1, y2+1):
                avg += I[y][x]
                pixel_count += 1

        avg /= pixel_count
        
        return avg

    def mean_binary(self, I, x1, y1, x2, y2):
        pixel_count = 0
        avg = 0
        for x in range(x1, x2+1):
            for y in range(y1, y2+1):
                if I[y][x]:
                    avg += 1
                pixel_count += 1

        avg /= pixel_count
        
        return avg
