import numpy as np

from PIL import Image

from tkinter.simpledialog import askfloat, askinteger
from tkinter import messagebox

import algo.utils as utils

from functools import partial

class Functions():
    def __init__(self, app_ref):
        self.app_ref = app_ref
    
    def negative_gray(self, I):
        for pixel in np.nditer(I, op_flags=['readwrite']):
            pixel[...] = np.subtract(255, pixel)
        
        return I

    def negative(self):
        I = self.app_ref.get_processed()

        fmt = utils.img_type(I)

        if fmt == 'RGB':
            I = utils.apply_gray_to_rgb(I, self.negative_gray)
        else:
            I = self.negative_gray(I)

        self.app_ref.set_processed(I)
    
    def multiply_by_scalar(self):
        scalar = askfloat("Multiplicación por escalar", "Escalar: ",
                  initialvalue=1,
                  minvalue=0,
                  maxvalue=255)

        I = self.app_ref.get_processed()

        for pixel in np.nditer(I, op_flags=['readwrite']):
            pixel[...] = np.multiply(scalar, pixel)

        self.app_ref.set_processed(I)

    def sum_other_image(self):
        self.apply_binary_op(lambda p1,p2: p1+p2)

    def substract_other_image(self):
        self.apply_binary_op(lambda p1,p2: p1-p2)

    def multiply_by_other_image(self):
        self.apply_binary_op(lambda p1,p2: p1*p2)

    def apply_binary_op(self, op):
        I1 = self.app_ref.get_processed()
        # load other image
        I2 = self.app_ref.load_image_from_file()

        # Perform function pixel by pixel
        for pix1, pix2 in np.nditer([I1, I2], op_flags=['readwrite']):
            pix1[...] = op(pix1, pix2)

        self.app_ref.set_processed(I1)

    def equalize_histogram(self):
        I = self.app_ref.get_processed()

        # Calculate histogram for image
        hist = utils.calculate_histogram(I, False)

        # Define cdf
        cdf = lambda k: hist[:k+1].sum()

        # Find cdfmin
        cdfmin = None
        for j in range(256):
            cdfmin_temp = cdf(j)
            if cdfmin_temp != 0:
                cdfmin = cdfmin_temp
                break
        
        # Count total pixels in image
        N = cdf(255)

        # Create transformation
        t = np.zeros(256)
        for j in range(256):
            t[j] = int((cdf(j)-cdfmin)/(N-cdfmin)*255)

        # Apply transformation
        for pixel in np.nditer(I, op_flags=['readwrite']):
            pixel[...] = t[pixel]

        self.app_ref.set_processed(I)

    def thresholding(self):
        thd = askinteger("Umbralizar", "Umbral: ", initialvalue = 127)

        I = self.app_ref.get_processed()

        for pixel in np.nditer(I, op_flags=['readwrite']):
            pixel[...] = 0 if pixel < thd else 255

        self.app_ref.set_processed(I)

    def gen_gauss(self, mu, desvio):
        return np.random.normal(mu, desvio)

    def gen_rayleigh(self, xhi):
        return np.random.rayleigh(xhi)

    def gen_exp(self, lamb):
        return np.random.exponential(1/lamb)
    
    def gen_uniform(self, min, max):
        return np.random.uniform(min, max)

    def noise_snp(self):
        p0 = askinteger("Ruido sal y pimienta", "Porcentaje inferior: ",
                    initialvalue=10)
        p1 = askinteger("Ruido sal y pimienta", "Porcentaje superior: ",
                    initialvalue=10)
        
        #iter over image
        I = self.app_ref.get_processed()

        for pixel in np.nditer(I, op_flags=['readwrite']):
            noised = self.gen_uniform(0,100)
            if noised <= p0:
                pixel[...] = 0
            elif noised >= (100-p1):
                pixel[...] = 255
            
            #ToDo for RGB

        self.app_ref.set_processed(I)

    def noise_additive_gauss(self):
        percentage = askinteger("Ruido gaussiano aditivo", "Porcentaje a contaminar: ",
                    initialvalue=10)
        mu = askfloat("Distribucion Gaussiana", "Variable μ: ",
                  initialvalue=0)
        desvio = askfloat("Distribucion Gaussiana", "Variable σ: ",
                  initialvalue=2)

        self.apply_noise(percentage, "gauss", "add", mu, desvio)
        
        print("Additive Gauss applied!")
        return 0
    
    def noise_multiplicative_rayleigh(self):
        percentage = askinteger("Ruido rayleigh multiplicativo", "Porcentaje a contaminar: ",
                    initialvalue=10)
        xhi = askfloat("Distribucion Rayleigh", "Variable ξ: ",
                  initialvalue=1)
        
        self.apply_noise(percentage, "rayleigh", "mul", xhi, None)

        print("Multiplicative Rayleigh applied!")
        return 0
    
    def noise_multiplicative_exp(self):
        percentage = askinteger("Ruido exponencial multiplicativo", "Porcentaje a contaminar: ",
                    initialvalue=30)
        lamb = askfloat("Distribucion Exponencial", "Variable λ: ",
                  initialvalue=1)

        self.apply_noise(percentage, "exp", "mul", lamb, None)

        print("Multiplicative exponential applied!")
        return 0

    def apply_noise(self, percentage, op_operation, op_type, var1, var2):
        #iter over image
        I = self.app_ref.get_processed()

        for pixel in np.nditer(I, op_flags=['readwrite']):
            noised = self.gen_uniform(0,100)
            if noised <= percentage:
                if op_operation == "exp":
                    random_number = self.gen_exp(var1)
                elif op_operation == "rayleigh":
                    random_number = self.gen_rayleigh(var1)
                elif op_operation == "gauss":
                    random_number = self.gen_gauss(var1, var2)
                
                if op_type == "mul":
                    pixel[...] = pixel*random_number
                elif op_type == "add":
                    pixel[...] = pixel+random_number
            
            #ToDo for RGB
        self.app_ref.set_processed(I)

    def mean_mask(self):
        self.mask(meanFilter)

    def gaussian_mask(self):
        stdv = askinteger("Filtro Gaussiano", "Valor del desvio estandar (σ): ", initialvalue = 1)
        gaussFilterWithSigma = lambda mask: gaussianFilter(mask, stdv)
        self.mask(gaussFilterWithSigma)

    def high_pass_mask(self):
        self.mask(highPassFilter)

    def median_mask(self):
        self.mask(np.median)

    def wmedian_mask(self):
        self.mask(weightedMedianFilter, 3)

    # maskFunc is the function that calculates
    # the value of the pixel based on the neighbor
    # pixels on the mask
    def mask(self, maskFunc, mask_dim=None, applying=True):
        if mask_dim == None:
            mask_dim = askinteger("Filtro de mascara", "Tamaño de la mascara (nxn): ", initialvalue = 3)

        I = self.app_ref.get_processed()
        
        if utils.img_type(I) == 'RGB':
            # split and apply mask
            R, G, B = utils.split_bands(I)
            I = utils.join_bands(
                self.mask_gray(R, maskFunc, mask_dim),
                self.mask_gray(G, maskFunc, mask_dim),
                self.mask_gray(B, maskFunc, mask_dim),
            )
        else:
            self.mask_gray(I, maskFunc, mask_dim)
        
        if applying:
            self.app_ref.set_processed(I)
        else:
            return I

    def mask_gray(self, I, maskFunc, mask_dim):
        #dependiendo del tipo de filtro hago un array con el peso correspondiente
        # lo vamos armando a medida que pasamos por los pixeles (algunos filtros
        # requieren saber el valor de los pixeles para armar la mascara)
        mask = np.zeros((mask_dim, mask_dim))

        #imagen a procesar
        #imagen que no cambia
        I_ref = np.copy(I)

        height, width = np.shape(I)
        for x in range(width):
            for y in range(height):
                for i in range(mask_dim):
                    for j in range(mask_dim):
                        coordx = x+i-np.floor(mask_dim/2)
                        coordy = y+j-np.floor(mask_dim/2)
                        if coordx < 0 or coordy < 0 or coordx >= width or coordy >= height:
                            #me fui entonces tengo que tomar una decision de las 4 propuestas
                            mask[i,j] = 0 # relleno con negro
                        else:
                            # estoy dentro de la imagen
                            # armo la mascara con los valores de los pixeles
                            # (despues llamo a la funcion para que me calcule el
                            # valor del pixel en base a sus vecinos)
                            mask[i, j] = I_ref[int(coordy), int(coordx)]

                
                #termine de armar la mascara, cambio el valor del pixel
                # llamo a la funcion que corresponda dependiendo del filtro
                I[y,x] = maskFunc(mask)

        return I

    def horizontal_border(self):
        self.mask(horizontal_filter)

    def vertical_border(self):
        self.mask(vertical_filter)
    
    def prewitt_border(self):
        filters = [horizontal_filter, vertical_filter]
        images = []
        for i in range(2):
            images.append(self.mask(filters[i], applying=False))

        self.sintetize(images)

    def alternative_border(self):
        images = []
        for i in range(4):
            images.append(self.mask(partial(rotative_filter, times = i), applying=False))

        self.sintetize(images)
    
    # Sintetizes one channel
    def sintetize_gray(self, images, sintetizer_form='max'):
        I = np.copy(images[0])
        height, width = np.shape(images[0])
        for x in range(width):
            for y in range(height):
                if sintetizer_form == 'max':
                    aux_pix = []
                    for img in images:
                        aux_pix.append(img[y,x])
                    I[y, x] = np.linalg.norm(aux_pix)
        	
        return I

    # if img is gray, sintetizes with sintetize_gray, if
    # img is rgb, sintetizes each channel with sintetize_gray
    def sintetize(self, images, sintetizer_form='norm'):
        if utils.img_type(images[0]) == 'RGB':
            Rs, Gs, Bs = [], [], []
            for I in images:
                R, G, B = utils.split_bands(I)
                Rs.append(R)
                Gs.append(G)
                Bs.append(B)

            I = utils.join_bands(
                self.sintetize_gray(Rs, sintetizer_form=sintetizer_form),
                self.sintetize_gray(Gs, sintetizer_form=sintetizer_form),
                self.sintetize_gray(Bs, sintetizer_form=sintetizer_form)
            )
        else:
            I = self.sintetize_gray(images, sintetizer_form=sintetizer_form)
        self.app_ref.set_processed(I)
    
def rotative_filter(mask, times=0):
    dim = mask.shape[0]

    weights = np.ones(mask.shape)
    mid = np.floor(dim/2)
    for y in range(dim):
        if y == mid:
            for x in range(dim):
                if x == mid:
                    weights[y,x] = -2
        elif y > mid:
            for x in range(dim):
                weights[y,x] = -1
    weights = utils.rotate_matrix3(weights,times)
    return np.sum(mask*weights)


#es el df/dy, supongo que es el vertical
def vertical_filter(mask):
    #creo que siempre son de 3x3, despues lo podemos cambiar para ser mas eficiente
    dim = mask.shape[0]

    weights = np.ones(mask.shape)
    mid_row = np.floor(dim/2)
    for y in range(dim):
        if y < mid_row:
            for x in range(dim):
                weights[y,x] = -1
        if y == mid_row:
            for x in range(dim):
                weights[y,x] = 0
        if y > mid_row:
            for x in range(dim):
                weights[y,x] = 1
    
    return np.sum(mask*weights)

#es el df/dx, supongo que es el horizontal
def horizontal_filter(mask):
    #creo que siempre son de 3x3, despues lo podemos cambiar para ser mas eficiente
    dim = mask.shape[0]

    weights = np.ones(mask.shape)
    mid_col = np.floor(dim/2)
    for x in range(dim):
        if x < mid_col:
            for y in range(dim):
                weights[y,x] = -1
        if x == mid_col:
            for y in range(dim):
                weights[y,x] = 0
        if x > mid_col:
            for y in range(dim):
                weights[y,x] = 1
    
    return np.sum(mask*weights)

# mask is the NxN submatrix
# of the image centered on the
# pixel
def meanFilter(mask):
    dim = mask.shape[0]
    # pixel weight is the same for all (1/N)
    weight = 1/dim**2

    return np.sum(mask*weight)

# stdv is sigma (not squared)
def gaussianFilter(mask, stdv):
    dim = mask.shape[0]

    # weights depend on position
    weights = np.ones(mask.shape)
    for x in range(dim):
        for y in range(dim):
            # remember (0,0) is the center
            relX = x-np.floor(dim/2)
            relY = y-np.floor(dim/2)

            weights[y,x] = (1/(2*np.pi*stdv**2))*np.exp(-(relX**2 + relY**2)/(stdv**2))
    
    weights = weights/np.sum(weights)
    
    return np.sum(mask*weights)

def highPassFilter(mask):
    dim = mask.shape[0]

    weights = np.ones(mask.shape) * -1;
    center = int(np.floor(dim/2))
    weights[center, center] = dim**2 - 1;
    weights /= dim**2;

    return np.sum(mask*weights)

def weightedMedianFilter(mask):
    if mask.shape[0] != 3:
        print("Im hardcoded for 3x3 masks!")
        return
    reps = np.array([[1,2,1], [2,4,2], [1,2,1]])
    arr = []
    for x in range(3):
        for y in range(3):
            for rep in range(reps[y,x]):
                arr.append(mask[y,x])

    return np.median(arr)