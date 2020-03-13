import tkinter as tk

class MenuBar(tk.Menu):
    def __init__(self, parent):
        tk.Menu.__init__(self, parent)

        # Create the menus

        ## File menu
        file_menu = tk.Menu(self, tearoff=0)
        self.add_cascade(label='Archivo', menu=file_menu)
        file_menu.add_command(label='Cargar...', command=parent.on_load_image)
        file_menu.add_command(label='Guardar...', command=parent.on_save_image)

        ## Generate menu
        gen_menu = tk.Menu(self, tearoff=0)
        self.add_cascade(label='Generar', menu=gen_menu)
        gen_menu.add_command(label='Cuadrado', command=parent.gen.square)
        gen_menu.add_command(label='Circulo', command=parent.gen.circle)
        gen_menu.add_command(label='Degrade de grises', command=parent.gen.gray_gradient)
        gen_menu.add_command(label='Degrade de colores', command=parent.gen.color_gradient)

        ## Test menu
        test_menu = tk.Menu(self, tearoff=0)
        self.add_cascade(label='Test', menu=test_menu)
        test_menu.add_command(label='Show selection info', command=parent.tools.test_selection)
        test_menu.add_command(label='Pintar seleccion', command=parent.tools.paint_selection)

