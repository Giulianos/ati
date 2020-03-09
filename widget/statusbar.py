import tkinter as tk

class StatusBar(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent, relief=tk.SUNKEN, bd=2)

        # Add current coordinates info
        self.coords_label = tk.Label(self, text='')
        self.coords_label.grid(row=0, column=0)

        # Add color info
        self.color_label = tk.Label(self, text='')
        self.color_label.grid(row=0, column=1)

    def set_coords(self, coords):
        self.coords_label.configure(text='Cursor: {}'.format(coords))

    def hide_coords(self):
        self.coords_label.configure(text='')
