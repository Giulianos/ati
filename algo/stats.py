import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import numpy as np
from PIL import Image
import tkinter as tk

from algo.utils import calculate_histogram
from algo.utils import img_type


class Stats():
    def __init__(self, app_ref):
        self.app_ref = app_ref

    def proc_pixel_value(self, x, y):
        I = self.app_ref.get_processed()
        color = I[y][x]

        if type(color) == np.bool_:
            return  1 if color else 0
        return color

    def region_mean_value(self):
        # Request region selection
        self.app_ref.mouse_selection.request_selection(self.handle_region_mean_value)

    def handle_region_mean_value(self, start, end):
        I = self.app_ref.get_processed()
        fmt = img_type(I)

        x_start, y_start = start
        x_end, y_end = end

        if fmt == 'BIN': # binary
            avg = self.mean_binary(I, x_start, y_start, x_end, y_end)
        elif fmt == 'GRAY': # grayscale
            avg = self.mean_grayscale(I, x_start, y_start, x_end, y_end)
        elif fmt == 'RGB': # rgb
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
    
    def show_histogram(self):
        # Create histograms
        I_orig = self.app_ref.get_original()
        hist_orig = calculate_histogram(I_orig)
        
        I_proc = self.app_ref.get_processed()
        hist_proc = calculate_histogram(I_proc)

        # Create plot window
        plot_window = tk.Toplevel()

        # Create figure
        fig = Figure(figsize=(10,4))
        ax_orig = fig.add_subplot(121)
        ax_proc = fig.add_subplot(122)

        ax_orig.plot(np.arange(256),hist_orig)
        ax_orig.set_title('Original')

        ax_proc.plot(np.arange(256),hist_proc)
        ax_proc.set_title('Procesada')

        # Create canvas in plot window
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        # canvas.show()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1.0)
        canvas.draw()



