import tkinter as tk

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
