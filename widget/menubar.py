import tkinter as tk

class MenuBar(tk.Menu):
    def __init__(self, parent):
        tk.Menu.__init__(self, parent)

        # Create the menus

        ## File menu
        file_menu = tk.Menu(self, tearoff=0)
        self.add_cascade(label='Archivo', menu=file_menu)
        file_menu.add_command(label='Cargar...', command=parent.on_load_image)

