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
        func_menu.add_command(label='Multiplicar por escalar', command=parent.funcs.multiply_by_scalar)
        func_menu.add_command(label='Sumar otra imagen', command=parent.funcs.sum_other_image)
        func_menu.add_command(label='Restar otra imagen', command=parent.funcs.substract_other_image)
        func_menu.add_command(label='Multiplicar por otra imagen', command=parent.funcs.multiply_by_other_image)
        func_menu.add_command(label='Ecualizar histograma', command=parent.funcs.equalize_histogram)
        func_menu.add_command(label='Umbralizar', command=parent.funcs.thresholding)

        func_menu.add_command(label='Pintar seleccion', command=parent.tools.paint_selection)
        func_menu.add_command(label='Cortar imagen', command=parent.tools.cut)

        ## Noise cascade in functions menu
        func_menu_ruido = tk.Menu(self, tearoff=0)
        func_menu.add_cascade(label='Agregar Ruido', menu=func_menu_ruido)
        func_menu_ruido.add_command(label='Gaussiano Aditivo', command=parent.funcs.noise_additive_gauss)
        func_menu_ruido.add_command(label='Raleigh Multiplicativo', command=parent.funcs.noise_multiplicative_raleigh)
        func_menu_ruido.add_command(label='Exponencial Multiplicativo', command=parent.funcs.noise_multiplicative_exp)
        func_menu_ruido.add_command(label='Sal y Pimienta', command=parent.funcs.noise_snp)

        ## Filter menu
        func_menu_filter = tk.Menu(self, tearoff=0)   
        func_menu.add_cascade(label='Agregar Filtro', menu=func_menu_filter)
        func_menu_filter.add_command(label='Filtro de la media', command=parent.funcs.mean_mask)     
        func_menu_filter.add_command(label='Filtro Gaussiano', command=parent.funcs.gaussian_mask)     
        func_menu_filter.add_command(label='Filtro Pasaaltos', command=parent.funcs.high_pass_mask)     
        func_menu_filter.add_command(label='Filtro de la mediana', command=parent.funcs.median_mask)     
        func_menu_filter.add_command(label='Filtro de la mediana ponderada', command=parent.funcs.wmedian_mask)     

        ## Info menu
        info_menu = tk.Menu(self, tearoff=0)
        self.add_cascade(label='Info', menu=info_menu)
        info_menu.add_command(label='Promedio de pixeles', command=parent.stats.region_mean_value)
        info_menu.add_command(label='Mostrar histograma', command=parent.stats.show_histogram)


