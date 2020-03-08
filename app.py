import tkinter as tk

from widget.imageviewer import ImageViewer

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        # Set window title
        self.title('ATI')

        # Add ImageViewer
        self.image_viewer = ImageViewer()
        self.image_viewer.pack()

