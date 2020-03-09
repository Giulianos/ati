import tkinter as tk

class StatusBar(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent, relief=tk.SUNKEN, bd=2)
        tk.Label(self, text="Click button to start process..").pack(fill=tk.X)
