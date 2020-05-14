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

    def thresholding(self, img=None, retrieveImg=True, ask=True, umbral=127, applying=True):
        if ask:
            thd = askinteger("Umbralizar", "Umbral: ", initialvalue = 127)
        else:
            thd = umbral
        if retrieveImg:
            I = self.app_ref.get_processed()
        else:
            I = img
        height, width = np.shape(I)

        for x in range(width):
            for y in range(height):
                I[y,x] = 0 if I[y,x] < thd else 255
        
        if applying:
            self.app_ref.set_processed(I)
        else:
            return I

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
    
    def bilateral_mask(self):
        ss = askfloat('Filtro Bilateral', 'σ espacial')
        sr = askfloat('Filtro Bilateral', 'σ color')
        bilateralFilterWithParams = lambda mask: bilateralFilter(mask, ss, sr)
        self.mask(bilateralFilterWithParams, mask_dim=int(2*ss+1))

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

    def sobel_border(self):
        filters = [sobel_horizontal_filter, sobel_vertical_filter]
        images = []
        for i in range(2):
            images.append(self.mask(filters[i], applying=False))

        self.sintetize(images)

    def alternative_border(self):
        images = []
        for i in range(4):
            images.append(self.mask(partial(rotative_filter, times = i), applying=False))

        self.sintetize(images)
    
    def laplace_border(self):
        self.mask(laplace_filter)
    
    def laplace_complete_border(self):
        #Preguntar si quiero max (OR) bordes o min (AND) bordes
        answer = messagebox.askyesno("Pregunta","Quiere maximizar los bordes encontrados?")
        input_umbral = askinteger("Cruce por 0", "Valor del umbral (u): ", initialvalue = 10)
        #aplico laplace
        self.mask(laplace_filter)
        

        # no usa umbral
        filters = [partial(horizontal_zero_check,u=input_umbral), partial(vertical_zero_check, u=input_umbral)]
        images = []
        for i in range(2):
            images.append(self.mask(filters[i], applying=False))


        self.sintetize(images, sintetizer_form=('or' if answer else 'and'))

    def laplace_gauss_border(self):
        #Preguntar si quiero max (OR) bordes o min (AND) bordes
        answer = messagebox.askyesno("Pregunta","Quiere maximizar los bordes encontrados?")
        input_sigma = askinteger("Filtro Gaussiano", "Valor del desvio estandar (σ): ", initialvalue = 1)
        input_umbral = askinteger("Cruce por 0", "Valor del umbral (u): ", initialvalue = 10)
        #aplico gauss
        self.mask(partial(gaussianFilter,stdv=input_sigma))
        #aplico laplace
        self.mask(laplace_filter)

        filters = [partial(horizontal_zero_check,u=input_umbral), partial(vertical_zero_check, u=input_umbral)]
        images = []
        for i in range(2):
            images.append(self.mask(filters[i], applying=False))

        self.sintetize(images, sintetizer_form=('or' if answer else 'and'))
    
    def canny_border(self):
        # 1. bilateral | podriamos modularizarlo aca
        #self.bilateral_mask()
        # 2. magnitud/modelo del gradiente de la img (genera M y Gx y Gy)
        #M = "sobel"
        # 3. calculo arctg(gy/gx) y discretizo el angulo y guardo en matriz
        #if Gx != 0:
        #    Angulos = arctg(Gy/Gx)
        #else:
        #    Angulos = 90
        #Si Angulos esta entre 0 y 22.5 o 157.5 a 180 --> 0
        #Si Angulos entre 22.5 y 67.5 --> 45
        #Si Angulos entre 67.5 y 112.5 --> 90
        #Si Angulos entre 112.5 y 157.5 --> 135
        # 4. supresion de no max (sobre M --> M1)
        #Por cada pixel, miro los adyacentes en su dir correspondiente y si alguno es mayor --> le pongo 0, sino le dejo su valor
        #Si son iguales, elijo
        # 5. umbralizacion con  histeresis (sobre M1)
        #tomo umbral con otsu vy estimo el desvio --> t1=t-desv t2=t+desv (t1<t2)
        #Elijo empezar horizontal o vertical
        #Los mayor t2 son borde
        #Los menor t1 no
        #Los del medio los tiro con respecto a sus adyacentes (0 o 255)
        return 0


    # Sintetizes one channel
    def sintetize_gray(self, images, sintetizer_form='norm'):
        I = np.copy(images[0])
        height, width = np.shape(images[0])
        for x in range(width):
            for y in range(height):
                aux_pix = []
                for img in images:
                    aux_pix.append(img[y,x])
                if sintetizer_form == 'norm':
                    I[y, x] = np.linalg.norm(aux_pix)
                elif sintetizer_form == 'or':
                    # hardcodeado para 2 imagenes
                    I[y, x] = (255 if aux_pix[0] == 255 or aux_pix[1] == 255 else 0) 
                elif sintetizer_form == 'and':
                    # hardcodeado para 2 imagenes
                    I[y, x] = (255 if aux_pix[0] == 255 and aux_pix[1] == 255 else 0)
        
        return I


    # if img is gray, sintetizes with sintetize_gray, if
    # img is rgb, sintetizes each channel with sintetize_gray
    def sintetize(self, images, sintetizer_form='norm', applying=True):
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
        
        if applying:
            self.app_ref.set_processed(I)
        else:
            return I

    def umbral_gobal(self):
        #iter over image
        I = self.app_ref.get_processed()

        u = np.mean(I)
        #umbralizo y separo en 2 arrays
        arr1 = []
        arr2 = []
        while True:
            for pixel in np.nditer(I, op_flags=['readwrite']):
                if pixel <= u:
                    arr1.append(pixel)
                else:
                    arr2.append(pixel)
            m1 = np.mean(arr1)
            m2 = np.mean(arr2)
            u_next = (m1+m2)/2
            if (u_next-u) <1:
                break
            else:
                u = u_next
                arr1.clear()
                arr2.clear()
        self.thresholding(ask=False, umbral=u)
        print("El umbral utilizado fue: "+ str(u))

    def umbral_otsu_wrap(self):
        I = self.app_ref.get_processed()
        if utils.img_type(I) == 'RGB':
            R, G, B = utils.split_bands(I)
            # ToDo: Resolver problema de join, dice que la np.shape(R) le devuelve mas de 2 cosas :S
            # creo que el problema lo genera esto que deforma el array en threashold for pixel in np.nditer(I, op_flags=['readwrite']):
            I = utils.join_bands(
                self.umbral_otsu(R),
                self.umbral_otsu(G),
                self.umbral_otsu(B)
            )
        else:
            I = self.umbral_otsu(I)
        
        self.app_ref.set_processed(I)


    def umbral_otsu(self, I):
        height, width = np.shape(I)
        #calculo l histograma normalizado
        unique, counts = np.unique(I, return_counts=True)
        counts = counts/(height*width)
        cumsum = []
        meansum = []
        #computo sumas acumuladas
        i = 0
        for count in counts:
            if i != 0:
                cumsum.append(cumsum[i-1]+count)
                meansum.append(meansum[i-1]+(count * unique[i]))
            else:
                cumsum.append(count)
                meansum.append(unique[i]*count)
            i += 1

        var = []
        i = 0
        for mean in meansum:
            if cumsum[i] != 1: 
                value = ((meansum[-1]*cumsum[i]-mean)**2)/(cumsum[i]*(1-cumsum[i]))
                var.append(value)
            
            i += 1

        result = np.where(var == np.amax(var))
        u = np.mean(result[0])
        print("El umbral calculado es: " + str(u))
        return self.thresholding(img=I, retrieveImg=False, ask=False, umbral=u, applying=False)
    
    def isotropic_difussion(self):
        # ToDo: ask user for parameters
        times = askinteger("Tiempo", "Tiempo de difusión: ", initialvalue = 1)

        I = self.app_ref.get_processed()
        g = lambda gradiente: 1

        if utils.img_type(I) == 'RGB':
            R, G, B = utils.split_bands(I)
            I = utils.join_bands(
                anisotropic_difussion(R, g, times, 0.25),
                anisotropic_difussion(G, g, times, 0.25),
                anisotropic_difussion(B, g, times, 0.25),
            )
        else:
            I = anisotropic_difussion(I, g, times, 0.25)

        self.app_ref.set_processed(I)
    
    def isotropic_difussion(self):
        # ToDo: ask user for parameters
        times = askinteger("Tiempo", "Tiempo de difusión: ", initialvalue = 1)

        I = self.app_ref.get_processed()
        g = lambda gradiente: 1

        if utils.img_type(I) == 'RGB':
            R, G, B = utils.split_bands(I)
            I = utils.join_bands(
                anisotropic_difussion(R, g, times, 0.25),
                anisotropic_difussion(G, g, times, 0.25),
                anisotropic_difussion(B, g, times, 0.25),
            )
        else:
            I = anisotropic_difussion(I, g, times, 0.25)

        self.app_ref.set_processed(I)
    
    def isotropic_difussion(self):
        self.difussion(lambda grad: 1)

    def anisotropic_leclerc_difussion(self):
        sigma = askinteger("Detector Leclerc", 'Parámetro sigma:', initialvalue = 0)
        leclerc_detector = lambda grad: np.exp(-1*(grad**2)/(sigma**2))
        self.difussion(leclerc_detector)
    
    def anisotropic_lorentziano_difussion(self):
        sigma = askinteger("Detector Leclerc", 'Parámetro sigma:', initialvalue = 0)
        lorentziano_detector = lambda grad: 1/((grad**2)/(sigma**2) + 1)
        self.difussion(lorentziano_detector)
    
    def difussion(self, g):
        # ToDo: ask user for parameters
        times = askinteger("Tiempo", "Tiempo de difusión: ", initialvalue = 1)

        I = self.app_ref.get_processed()

        if utils.img_type(I) == 'RGB':
            R, G, B = utils.split_bands(I)
            I = utils.join_bands(
                anisotropic_difussion(R, g, times, 0.25),
                anisotropic_difussion(G, g, times, 0.25),
                anisotropic_difussion(B, g, times, 0.25),
            )
        else:
            I = anisotropic_difussion(I, g, times, 0.25)

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

def laplace_filter(mask):
    #creo que siempre son de 3x3, despues lo podemos cambiar para ser mas eficiente
    dim = mask.shape[0]

    weights = np.zeros(mask.shape)
    mid = np.floor(dim/2)
    for y in range(dim):
        for x in range(dim):
            if y != mid:
                if x == mid:
                    weights[y,x] = -1
            else:
                if x == mid:
                    weights[y,x] = 4
                else:
                    weights[y,x] = -1
    
    return np.sum(mask*weights)

def vertical_zero_check(mask, u=0):
    #creo que siempre son de 3x3, despues lo podemos cambiar para ser mas eficiente
    dim = mask.shape[0]
    mid = int(np.floor(dim/2))
    
    #ToDo: Hay que agregar el umbral para la pendiente |a|+|b| > U
    mid_pix = mask[mid,mid]
    next_pix = mask[mid+1, mid]
    if mid_pix * next_pix < 0:
        return 255 if abs(mid_pix) + abs(next_pix) > u else 0
    # aca me fijo si estoy en un caso de pixel 0, por lo que evaluo atras y adelante
    # (Habria que verificar que no nos comemos ningun caso especial)
    elif mask[mid,mid] == 0:
        prev_pix = mask[mid-1, mid]
        if prev_pix * next_pix < 0:
            return 255 if abs(prev_pix) + abs(next_pix) > u else 0
        else:
            return 0
    else:
        return 0

def horizontal_zero_check(mask, u=0):
    #creo que siempre son de 3x3, despues lo podemos cambiar para ser mas eficiente
    dim = mask.shape[0]
    mid = int(np.floor(dim/2))
    
    #ToDo: Hay que agregar el umbral para la pendiente |a|+|b| > U
    mid_pix = mask[mid,mid]
    next_pix = mask[mid, mid+1]
    if mid_pix * next_pix < 0:
        return 255 if abs(mid_pix) + abs(next_pix) > u else 0
    # aca me fijo si estoy en un caso de pixel 0, por lo que evaluo atras y adelante
    # (Habria que verificar que no nos comemos ningun caso especial)
    elif mask[mid,mid] == 0:
        prev_pix = mask[mid, mid-1]
        if prev_pix * next_pix < 0:
            return 255 if abs(prev_pix) + abs(next_pix) > u else 0
        else:
            return 0
    else:
        return 0

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

def sobel_horizontal_filter(mask):
    dim = mask.shape[0]

    weights = np.ones(mask.shape)
    values = [-1,0,1,-2,0,2,-1,0,1]
    for i in range(dim):
        for j in range(dim):
            weights[i,j] = values[(i*dim)+j]

    return np.sum(mask*weights)

def sobel_vertical_filter(mask):
    dim = mask.shape[0]

    weights = np.ones(mask.shape)
    values = [-1,-2,-1,0,0,0,1,2,1]

    for i in range(dim):
        for j in range(dim):
            weights[i,j] = values[(i*dim)+j]

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

# Returns N,S,E,W neighbors for pixel
# (x,y) in img
def cardinal_neighbors(img, x, y):
    height, width = np.shape(img)[0:2]
    neighbors = []
    neighbors.append(img[y,x + 1] if x + 1 < width else img[y,x])
    neighbors.append(img[y,x - 1] if x - 1 >= 0 else img[y,x])
    neighbors.append(img[y + 1,x] if y + 1 < height else img[y,x])
    neighbors.append(img[y - 1,x] if y - 1 >= height else img[y,x])

    return neighbors

# generic anisotropic difussion (c is the border function)
def anisotropic_difussion(img, g, times, lambda_param):
    # It (current)
    img_curr = np.array(img)
    # It+1 (next)
    img_next = np.array(img)

    height, width = np.shape(img)[0:2]
    for t in range(times):
        for x in range(width):
            for y in range(height):
                sum_next = 0
                for n in cardinal_neighbors(img_curr, x, y):
                    d = n - img_curr[y,x]
                    sum_next += d * g(d)
                img_next[y,x] = img_curr[y,x] + sum_next*lambda_param
        # now replace current with next to start next iteration
        img_curr = img_next
    
    print(np.shape(img_curr))

    return img_curr

def bilateralFilter(mask, ss, sr):
    omega = lambda i,j,k,l: np.exp(
        -1*(((i-k)**2 + (j-l)**2) / (2*(ss**2))) -
       ((mask[i,j]-mask[k,l])**2 / (2*(sr**2))) 
    )

    rows, cols = np.shape(mask)[0:2]
    i, j = int(rows/2), int(cols/2)
    denom_sum = 0
    num_sum = 0
    for k in range(rows):
        for l in range(cols):
            if (i,j) != (k,l):
                num_sum += mask[k,l]*omega(i,j,k,l)
                denom_sum += omega(i,j,k,l)

    return num_sum/denom_sum


