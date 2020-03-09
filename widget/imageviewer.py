import tkinter as tk
from PIL import ImageTk

class ImageViewer(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent, bd=2, relief=tk.SUNKEN)

        # Create image scrolling frame
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        xscroll = tk.Scrollbar(self, orient=tk.HORIZONTAL)
        xscroll.grid(row=1, column=0, sticky=tk.E+tk.W)
        yscroll = tk.Scrollbar(self)
        yscroll.grid(row=0, column=1, sticky=tk.N+tk.S)
        self.canvas = tk.Canvas(self, width=500, height=500, bd=0, highlightthickness=0, relief='ridge', xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)

        self.canvas.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        xscroll.config(command=self.canvas.xview)
        yscroll.config(command=self.canvas.yview)
        
        # Bind mouse event handler
        self.canvas.bind('<Motion>', self.on_mouse_move)
        self.mouse_handler = None

    def set_image(self, pil_img):
        # Create the image in the canvas
        self.photo_image = ImageTk.PhotoImage(pil_img)
        self.canvas.create_image(0,0, anchor='nw', image=self.photo_image)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

    def set_mouse_handler(self, mouse_handler):
        self.mouse_handler = mouse_handler

    def on_mouse_move(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)

        self.canvas.configure(cursor='crosshair')

        if self.mouse_handler != None:
            self.mouse_handler(self, x, y)
