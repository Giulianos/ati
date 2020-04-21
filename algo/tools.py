import numpy as np

from tkinter.simpledialog import askfloat

from PIL import Image

import algo.utils as utils

class Tools():
    def __init__(self, app_ref):
        self.app_ref = app_ref

    def test_selection(self):
       print('Requested selection') 
       # Here we request a selection that will be handled
       # by test_selection_handler
       self.app_ref.mouse_selection.request_selection(self.test_selection_handler)

    # This is called once we have the selection
    def test_selection_handler(self, start, end):
       # Here we do something with the selection
       print('Successfully selected from {} to {}!'.format(start, end))

    def paint_selection(self):
        self.app_ref.mouse_selection.request_selection(self.paint_selection_handler)
    
    
        
    def paint_selection_handler(self, start, end):
        I = self.app_ref.get_processed()
        x_start, y_start = start
        x_end, y_end = end
        
        def paint_grayscale(I):
            for x in range(x_start, x_end+1):
                for y in range(y_start, y_end+1):
                    I[y][x] = 0
            return I   

        if utils.img_type(I) == 'RGB':
            utils.apply_gray_to_rgb(I, paint_grayscale)
        else:
            paint_grayscale(I)

        self.app_ref.set_processed(I)


    def cut(self):
        self.app_ref.mouse_selection.request_selection(self.handle_cut)

    def handle_cut(self, start, end):
        I = self.app_ref.get_processed()
        
        x1, y1 = start
        x2, y2 = end

        self.app_ref.set_processed(I[y1:y2, x1:x2])
    
    def dinamic_range_compression(self):
        I = self.app_ref.get_processed()

        max_val = np.max(I)

        c_val = (255 - 1)/np.log(1+max_val)

        for pixel in np.nditer(I, op_flags=['readwrite']):
            pixel[...] = c_val*np.log(1+pixel) 

        self.app_ref.set_processed(I)

    def gamma_power(self):
        #permite elegir 0 y 2, aunque el metodo no lo permita
        gamma = askfloat("Modificar Contraste", "Variable γ:",
                    initialvalue=1,  minvalue=0.0, maxvalue=2.0)

        c_val = np.power((255-1),(1-gamma))

        I = self.app_ref.get_processed()

        for pixel in np.nditer(I, op_flags=['readwrite']):
            pixel[...] = c_val*np.power(pixel,gamma)

        self.app_ref.set_processed(I)

