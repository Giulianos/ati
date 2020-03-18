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

        ## Functions menu
        func_menu = tk.Menu(self, tearoff=0)
        self.add_cascade(label='Funciones', menu=func_menu)
        func_menu.add_command(label='Negativo', command=parent.funcs.negative)
        func_menu.add_command(label='Sumar otra imagen', command=parent.funcs.sum_other_image)
        func_menu.add_command(label='Restar otra imagen', command=parent.funcs.substract_other_image)
        func_menu.add_command(label='Multiplicar por otra imagen', command=parent.funcs.multiply_by_other_image)
        func_menu.add_command(label='Ecualizar histograma', command=parent.funcs.equalize_histogram)

        func_menu.add_command(label='Pintar seleccion', command=parent.tools.paint_selection)
        func_menu.add_command(label='Cortar imagen', command=parent.tools.cut)

        ## Info menu
        info_menu = tk.Menu(self, tearoff=0)
        self.add_cascade(label='Info', menu=info_menu)
        info_menu.add_command(label='Promedio de pixeles', command=parent.stats.region_mean_value)
        info_menu.add_command(label='Mostrar histograma', command=parent.stats.show_histogram)


