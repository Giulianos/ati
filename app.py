import tkinter as tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
from PIL import Image

from widget.imageviewer import ImageViewer
from widget.menubar import MenuBar

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

        # Add ImageViewer
        self.image_viewer = ImageViewer()
        self.image_viewer.pack()

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
        self.image_viewer.set_original(self.img_orig)
        self.image_viewer.set_processed(self.img_proc)


