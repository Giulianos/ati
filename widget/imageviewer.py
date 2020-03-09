import tkinter as tk
from PIL import ImageTk

class ImageViewer(tk.Frame):
    def __init__(self, label):
        tk.Frame.__init__(self)

        # Create image scrolling frame
        frame = tk.Frame(self, bd=2, relief=tk.SUNKEN)
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        
        xscroll = tk.Scrollbar(frame, orient=tk.HORIZONTAL)
        xscroll.grid(row=1, column=0, sticky=tk.E+tk.W)
        yscroll = tk.Scrollbar(frame)
        yscroll.grid(row=0, column=0, sticky=tk.N+tk.S)
        self.canvas = tk.Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
        self.canvas.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        xscroll.config(command=self.canvas.xview)
        yscroll.config(command=self.canvas.yview)
        frame.pack(fill=tk.BOTH, expand=1)


    def set_image(self, pil_img):
        # Create the image in the canvas
        self.photo_image = ImageTk.PhotoImage(pil_img)
        self.canvas.create_image(0,0, anchor='nw', image=self.photo_image)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
