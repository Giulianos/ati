import tkinter as tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter.simpledialog import askinteger
from PIL import Image

from widget.imageviewer import ImageViewer
from widget.menubar import MenuBar
from widget.statusbar import StatusBar

from algo.gen import Gen
from algo.stats import Stats
from algo.tools import Tools
from algo.functions import Functions

from mouse import MouseSelection

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        # Set window title
        self.title('ATI')
        
        # Add the Gen object (to generate images)
        self.gen = Gen(self)

        # Add Stats object (to get stats of images)
        self.stats = Stats(self)

        # Add Tools object (to manipulate images)
        self.tools = Tools(self)

        self.funcs = Functions(self)

        # Add MouseSelection object (to handle requests for mouse selections)
        self.mouse_selection = MouseSelection()
        
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
        self.iv_proc.set_mouse_handler(self.on_proc_mouse_move)
        self.iv_proc.set_mouse_leave_handler(self.on_proc_mouse_leave)
        self.iv_proc.set_mouse_selection(self.mouse_selection)

        # Add StatusBar
        self.statusbar = StatusBar(self)
        self.statusbar.grid(row=2, column=0, columnspan=2, sticky=tk.E+tk.W)

    def load_image_from_file(self):
        # Get the image path from the dialog
        # TODO: add RAW filetype (requires special opening)
        image_path = askopenfilename(filetypes=[('PPM', '.ppm'), ('PGM', '.pgm'), ('RAW', '.raw')])

        # Open the image from the file
        if ".RAW" in image_path or ".raw" in image_path:
            w = askinteger("Raw Image", "Width: ",
                  minvalue=2,
                  maxvalue=1000)
            h = askinteger("Raw Image", "Height: ",
                  minvalue=2,
                  maxvalue=1000)
            with open(image_path, "rb") as binary_file:
                databytes = binary_file.read()
                img = Image.frombytes("L", (w,h), databytes, decoder_name='raw')
        else: 
            img = Image.open(image_path)

        return img

    def on_load_image(self):
        # Load into app
        self.set_original(self.load_image_from_file())

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

    # Sets the processed image (use this to apply any modification to the image)
    def set_processed(self, img):
        self.img_proc = img
        self.iv_proc.set_image(self.img_proc)

    # Mouse handler for processed image
    def on_proc_mouse_move(self, canvas, x, y):

        try:
            # Get pixel color in mouse position
            color = self.stats.proc_pixel_value(x, y)

            # Show color and coordinates in statusbar
            self.statusbar.set_color(color)
            self.statusbar.set_coords((x,y))
        except:
            self.statusbar.hide_coords()
            self.statusbar.hide_color()

    def on_proc_mouse_leave(self, event):
        self.statusbar.hide_coords()
        self.statusbar.hide_color()

