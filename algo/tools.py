import numpy as np

from tkinter.simpledialog import askfloat

from PIL import Image

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
        img = self.app_ref.img_proc
        I = np.array(img)
        x_start, y_start = start
        x_end, y_end = end
        for x in range(x_start, x_end+1):
            for y in range(y_start, y_end+1):
                if img.getbands() == ('1',): # image is binary
                    I[y][x] = False
                elif img.getbands() == ('L',): # image is grayscale
                    I[y][x] = 0
                elif img.getbands() == ('R','G','B'):
                    I[y][x] = (0,0,0)

        # Convert image back to PIL
        img = Image.fromarray(I)
        self.app_ref.set_processed(img)


    def cut(self):
        self.app_ref.mouse_selection.request_selection(self.handle_cut)

    def handle_cut(self, start, end):
        img = self.app_ref.img_proc
        
        x1, y1 = start
        x2, y2 = end

        img = img.crop((x1,y1,x2,y2))

        self.app_ref.set_processed(img)
    
    def dinamic_range_compression(self):
        img = self.app_ref.img_proc

        bands = img.getbands()
        if bands == ('1',):
            img = img.convert('L')
        
        min_val, max_val = img.getextrema()
        c_val = (255 - 1)/np.log(1+max_val)

        I = np.array(img)    
        
        for pixel in np.nditer(I, op_flags=['readwrite']):
            pixel[...] = c_val*np.log(1+pixel) 

        img = Image.fromarray(I)
        if bands == ('1',):
            img = img.convert('1')
        self.app_ref.set_processed(img)

    def gamma_power(self):
        img = self.app_ref.img_proc

        #permite elegir 0 y 2, aunque el metodo no lo permita
        gamma = askfloat("Modificar Contraste", "Variable γ:",
                    initialvalue=1,  minvalue=0.0, maxvalue=2.0)

        c_val = np.power((255-1),(1-gamma))

        bands = img.getbands()
        if bands == ('1',):
            img = img.convert('L')
        
        I = np.array(img)    
        
        for pixel in np.nditer(I, op_flags=['readwrite']):
            pixel[...] = c_val*np.power(pixel,gamma)

        img = Image.fromarray(I)
        if bands == ('1',):
            img = img.convert('1')
        self.app_ref.set_processed(img)

