import tkinter as tk
from PIL import Image

from widget.imageviewer import ImageViewer

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        # Set window title
        self.title('ATI')

        # Add ImageViewer
        self.image_viewer = ImageViewer()
        self.image_viewer.pack()

        # Add test buttons to load images
        tk.Button(self, text='Cargar original', command=self.on_load_original).pack()
        tk.Button(self, text='Cargar procesado', command=self.on_load_processed).pack()

    def on_load_original(self):
        # Open the image from the file
        self.img_orig = Image.open('Lenaclor.ppm')

        # Set the image in the ImageViewer
        self.image_viewer.set_original(self.img_orig)

    def on_load_processed(self):
        # Open the image from the file
        self.img_proc = Image.open('Lenaclor.ppm')

        # Set the image in the ImageViewer
        self.image_viewer.set_processed(self.img_proc)
