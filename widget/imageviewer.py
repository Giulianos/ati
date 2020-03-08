import tkinter as tk
from PIL import ImageTk

class ImageViewer(tk.Frame):
    def __init__(self):
        tk.Frame.__init__(self)

        # Add image labels
        tk.Label(self, text='Original').grid(row=0, column=0)
        tk.Label(self, text='Procesado').grid(row=0, column=1)
        
        # Add images placeholder (save as class prop to load the images later)
        self.canvas_orig = tk.Canvas(self, width=600, height=600, bg='grey')
        self.canvas_orig.grid(row=1, column=0)
        self.canvas_proc = tk.Canvas(self, width=600, height=600, bg='grey')
        self.canvas_proc.grid(row=1, column=1)

    def set_original(self, pil_img):
        # Get canvas dimensions
        canvasw = self.canvas_orig.winfo_width()
        canvash = self.canvas_orig.winfo_height()

        # Create the image in the canvas
        self.photo_orig = ImageTk.PhotoImage(pil_img)
        self.canvas_orig.create_image(canvasw/2, canvash/2, image=self.photo_orig)
        
    def set_processed(self, pil_img):
        # Get canvas dimensions
        canvasw = self.canvas_proc.winfo_width()
        canvash = self.canvas_proc.winfo_height()

        # Create the image in the canvas
        self.photo_proc = ImageTk.PhotoImage(pil_img)
        self.canvas_proc.create_image(canvasw/2, canvash/2, image=self.photo_proc)
