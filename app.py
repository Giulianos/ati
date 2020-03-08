import tkinter as tk
from tkinter.filedialog import askopenfilename
from PIL import Image

from widget.imageviewer import ImageViewer
from widget.menubar import MenuBar

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        # Set window title
        self.title('ATI')

        # Add MenuBar
        menubar = MenuBar(self)
        self.config(menu=menubar)

        # Add ImageViewer
        self.image_viewer = ImageViewer()
        self.image_viewer.pack()
        
    def on_load_image(self):
        # Get the image path from the dialog
        # TODO: add RAW filetype (requires special opening)
        image_path = askopenfilename(filetypes=[('PPM', '.ppm'), ('PGM', '.pgm')])

        # Open the image from the file
        self.img_orig = Image.open(image_path)
        self.img_proc = Image.open(image_path)

        # Set the images in the ImageViewer
        self.image_viewer.set_original(self.img_orig)
        self.image_viewer.set_processed(self.img_proc)
