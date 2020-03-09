import tkinter as tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
from PIL import Image

from widget.imageviewer import ImageViewer
from widget.menubar import MenuBar
from widget.statusbar import StatusBar

from algo.gen import Gen

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        # Set window title
        self.title('ATI')
        
        # Add the Gen object (to generate images)
        self.gen = Gen(self)
        
        # Add MenuBar
        menubar = MenuBar(self)
        self.config(menu=menubar)

        # Configure grid
        tk.Grid.rowconfigure(self, 1, weight=1)
        tk.Grid.columnconfigure(self, 0, weight=1)
        tk.Grid.columnconfigure(self, 1, weight=1)

        # Add ImageViewer for original
        self.iv_orig = ImageViewer(self)
        self.iv_orig.grid(row=1, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        tk.Label(self, text='Original').grid(row=0, column=0)

        # Add ImageViewer for processed
        self.iv_proc = ImageViewer(self)
        self.iv_proc.grid(row=1, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        tk.Label(self, text='Procesada').grid(row=0, column=1)

    def on_load_image(self):
        # Get the image path from the dialog
        # TODO: add RAW filetype (requires special opening)
        image_path = askopenfilename(filetypes=[('PPM', '.ppm'), ('PGM', '.pgm')])

        # Open the image from the file
        img = Image.open(image_path)

        # Load into app
        self.set_original(img)

    def on_save_image(self):
        # Get the path were the image will be saved
        image_path = asksaveasfilename(filetypes=[
            ('PPM', '.ppm'),
            ('PGM', '.pgm'),
            ('PNG', '.png'),
        ])

        # Save using PIL (format is inferred from extension)
        self.img_proc.save(image_path)

    # Sets the original image (setting also the processed)
    def set_original(self, img):
        self.img_orig = img
        self.img_proc = img.copy()

        # Set the images in the ImageViewer
        self.iv_orig.set_image(self.img_orig)
        self.iv_proc.set_image(self.img_proc)


