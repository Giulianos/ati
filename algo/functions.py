import numpy as np
import math as mt
import heapq as hpq

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

        for y in range(height):
            for x in range(width):
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

    def bilateral_mask(self, ss=None, sr=None, applying=True):
        if ss == None:
            ss = askfloat('Filtro Bilateral', 'σ espacial')
        if sr == None:
            sr = askfloat('Filtro Bilateral', 'σ color')
        bilateralFilterWithParams = lambda mask: bilateralFilter(mask, ss, sr)
        if applying:
            self.mask(bilateralFilterWithParams, mask_dim=int(2*ss+1))
        else:
            return self.mask(bilateralFilterWithParams, mask_dim=int(2*ss+1), applying=False)

    # maskFunc is the function that calculates
    # the value of the pixel based on the neighbor
    # pixels on the mask
    def mask(self, maskFunc, img=None, retrieveImg=True, mask_dim=None, applying=True):
        if mask_dim == None:
            mask_dim = askinteger("Filtro de mascara", "Tamaño de la mascara (nxn): ", initialvalue = 3)
        if retrieveImg:
            I = self.app_ref.get_processed()
        else:
            I = img

        if utils.img_type(I) == 'RGB':
            # split and apply mask
            R, G, B = utils.split_bands(I)
            I = utils.join_bands(
                self.mask_gray(R, maskFunc, mask_dim),
                self.mask_gray(G, maskFunc, mask_dim),
                self.mask_gray(B, maskFunc, mask_dim),
            )
        else:
            I = self.mask_gray(I, maskFunc, mask_dim)

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
        for y in range(height):
            for x in range(width):
                for j in range(mask_dim):
                    for i in range(mask_dim):
                        coordx = x+i-np.floor(mask_dim/2)
                        coordy = y+j-np.floor(mask_dim/2)
                        if coordx < 0 or coordy < 0 or coordx >= width or coordy >= height:
                            #me fui entonces tengo que tomar una decision de las 4 propuestas
                            mask[j,i] = 0 # relleno con negro
                        else:
                            # estoy dentro de la imagen
                            # armo la mascara con los valores de los pixeles
                            # (despues llamo a la funcion para que me calcule el
                            # valor del pixel en base a sus vecinos)
                            mask[j, i] = I_ref[int(coordy), int(coordx)]


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

    def sobel_border(self, applying=True, canny=False, img=None):
        filters = [sobel_horizontal_filter, sobel_vertical_filter]
        images = []
        for i in range(2):
            images.append(self.mask(filters[i], applying=False, img=img))

        if applying:
            self.sintetize(images)
        elif canny:
            images.append(self.sintetize(images, applying=False))
            return images
        else:
            return self.sintetize(images, applying=False)


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

    def variableMask(self, mask, dir):
        dim = mask.shape[0]
        mid = int(np.floor(dim/2))
        #Estoy tomando como mayor estricto asi que si son iguales tomo como que no es borde. Eso puede estar mal
        #print(dir)
        if dir == 0:
            return mask[mid,mid] if mask[mid,mid] > mask[mid,mid+1] and mask[mid,mid] > mask[mid,mid-1] else 0
        elif dir == 45:
            return mask[mid,mid] if mask[mid,mid] > mask[mid+1,mid+1] and mask[mid,mid] > mask[mid-1,mid-1] else 0
        elif dir == 90:
            return mask[mid,mid] if mask[mid,mid] > mask[mid+1,mid] and mask[mid,mid] > mask[mid-1,mid] else 0
        elif dir == 135:
            return mask[mid,mid] if mask[mid,mid] > mask[mid-1,mid+1] and mask[mid,mid] > mask[mid+1,mid-1] else 0

    def thresholding_conexo_8(self, mask, u1, u2):
        dim = mask.shape[0]
        mid = int(np.floor(dim/2))
        value_mid = mask[mid,mid]
        if value_mid < u1:
            return 0
        elif value_mid > u2:
            return 255
        else:
            conexo = 255 in mask
            return 255 if conexo else 0

    def thresholding_conexo_4(self, mask, u1, u2):
        dim = mask.shape[0]
        mid = int(np.floor(dim/2))
        value_mid = mask[mid,mid]
        values = [mask[mid+1,mid], mask[mid,mid+1], mask[mid-1,mid], mask[mid,mid-1]]
        if value_mid < u1:
            return 0
        elif value_mid >= u2:
            return 255
        else:
            conexo = 255 in values
            return 255 if conexo else 0

    def canny_border(self):
        # 1. bilateral | podriamos modularizarlo aca
        bilateral_image = self.bilateral_mask(ss=2,sr=30, applying=False)
        # 2. magnitud/modelo del gradiente de la img (genera M y Gx y Gy)
        border_imgs = self.sobel_border(applying=False, canny=True, img=bilateral_image)

        #opcion1
        M = border_imgs[2]
        Gy = border_imgs[1]
        Gx = border_imgs[0]

        #opcion2
        # M = border_imgs[2]
        # Gx = border_imgs[1]
        # Gy = border_imgs[0]

        # 3. calculo arctg(gy/gx) y discretizo el angulo y guardo en matriz
        dir = np.copy(M)
        height, width = np.shape(M)
        for y in range(height):
            for x in range(width):
                if Gx[y,x] == 0:
                    if Gy[y,x] != 0:
                        dir[y,x] = 90
                    else:
                        dir[y,x] = 0
                else:
                    partialDeg = np.rad2deg(mt.atan2(Gy[y,x],Gx[y,x]))
                    partialDeg = partialDeg if partialDeg>=0 else 180+partialDeg
                    #print(partialDeg)
                    if partialDeg >= 0 and partialDeg < 22.5 or partialDeg > 157.5 and partialDeg <= 180:
                        dir[y, x] = 0
                    elif partialDeg >= 22.5 and partialDeg < 67.5:
                        dir[y, x] = 45
                    elif partialDeg >= 67.5 and partialDeg < 112.5:
                        dir[y, x] = 90
                    elif partialDeg >= 112.5 and partialDeg <= 157.5:
                        dir[y, x] = 135
        np.savetxt("dir.txt", dir, fmt="%s")
        np.savetxt("sobel.txt", M, fmt="%s")
        # 4. supresion de no max (sobre M --> M1)
        #Por cada pixel, miro los adyacentes en su dir correspondiente y si alguno es mayor --> le pongo 0, sino le dejo su valor
        #Si son iguales, elijo
        M1 = np.copy(M)
        mask_dim = 3
        mask = np.zeros((mask_dim, mask_dim))
        #Hago lo mismo que en mask pero tengo que usar una funcion que me permita cambiar la mascara a partir de la direccion del borde
        for y in range(height):
            for x in range(width):
                if M1[y,x] != 0:
                    for j in range(mask_dim):
                        for i in range(mask_dim):
                            coordx = x+i-np.floor(mask_dim/2)
                            coordy = y+j-np.floor(mask_dim/2)
                            if coordx < 0 or coordy < 0 or coordx >= width or coordy >= height:
                                mask[j,i] = 0
                            else:
                                mask[j, i] = M1[int(coordy), int(coordx)]

                    M1[y,x] = self.variableMask(mask, dir[y,x])

        np.savetxt("supresion.txt", M1, fmt="%s")
        # 5. umbralizacion con  histeresis (sobre M1)
        #tomo umbral con otsu vy estimo el desvio --> t1=t-desv t2=t+desv (t1<t2)
        #ojo
        ret = self.umbral_otsu(M1,applying=False)
        t1 = ret[0] - ret[1]
        print(t1)
        # t1 = 50
        t2 = ret[0] + ret[1]
        print(t2)
        # t2 = 100
        #Aca hay que ver si usamos iRef o no, y sobre que imagen calculamos el umbral.
        iRef = np.copy(M1)
        iFinal = np.copy(M1)
        for y in range(height):
            for x in range(width):
                for j in range(mask_dim):
                    for i in range(mask_dim):
                        coordx = x+i-np.floor(mask_dim/2)
                        coordy = y+j-np.floor(mask_dim/2)
                        if coordx < 0 or coordy < 0 or coordx >= width or coordy >= height:
                            mask[j,i] = 0
                        else:
                            mask[j, i] = iRef[int(coordy), int(coordx)]

                iFinal[y,x] = self.thresholding_conexo_4(u1=t1, u2=t2, mask=mask)

        self.app_ref.set_processed(iFinal)




    # Sintetizes one channel
    def sintetize_gray(self, images, sintetizer_form='norm'):
        I = np.copy(images[0])
        height, width = np.shape(images[0])
        for y in range(height):
            for x in range(width):
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


    def umbral_otsu(self, I, applying=True):
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
        maxvar = np.amax(var)
        result = np.where(var == maxvar)
        u = np.mean(result[0])
        print("El umbral calculado es: " + str(u))
        if applying:
            return self.thresholding(img=I, retrieveImg=False, ask=False, umbral=u, applying=False)
        else:
            return [u,maxvar**(1/2)]

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

    def susan(self, borders=True, corners=True):
        threshold = askinteger("Umbral de color", "Umbral de color: ", initialvalue=27)

        I = self.app_ref.get_processed()

        if utils.img_type(I) == 'RGB':
            # TODO: implementar para rgb
            return

        img = susan(np.array(I), threshold, borders, corners)

        self.app_ref.set_processed(Image.fromarray(img))

    lin = []
    lout = []
    phi = None
    img = None
    object_avg = None
    background_avg = None

    # Flujo paso 1
    def contornos_activos_trigger(self):
        #PASOS
        #1 Selecciono region rectangular y defino LIN y LOUT
        self.semaphore = False
        self.app_ref.mouse_selection.request_selection(self.selection_prep)


    def contornos_activos_wrap(self, lin, lout, phi,width,height):
        iterations = 0
        maxiterations = 400
        nothing_else = False
        #repito el ciclo hasta que no quede ningun cambio o me quede sin iteraciones
        while nothing_else == False and iterations < maxiterations:
            ret_values = self.contornos_cicle(lin, lout, phi,width,height)
            nothing_else = ret_values[0]
            lin = ret_values[1]
            lout = ret_values[2]
            phi = ret_values[3]
            iterations += 1

        #voy a tener una matrix que tenga 3 fondo 1 borde out y -1 borde in -3 objeto
        return phi

    def update_avgs(self):
        height, width = np.shape(self.phi)
        obj_sum = 0
        bg_sum = 0
        obj_n = 0
        bg_n = 0
        for y in range(height):
            for x in range(width):
                if self.phi[y,x] == -3:
                    obj_sum += self.img[y, x]
                    obj_n += 1
                elif self.phi[y,x] == 3:
                    bg_sum += self.img[y, x]
                    bg_n += 1
        self.object_avg = obj_sum/obj_n if obj_n > 0 else 0
        self.background_avg = bg_sum/bg_n if bg_n > 0 else 0

    def contornos_cicle(self, lin, lout, phi, width, height):
        nothing_else = True
        #2 Para cada LOUT si Fd(x)>0 entonces borro x de LOUT y lo agrego a LIN.
        for point in lout:
            #Evalua Fd(x) = log(Norma(caracteristicasfond(x)-caracteristicaspixel(x))/norma(caracteristicasobjeto(x)-caracteristicaspixel(x)))
            #si f(x)<0 --> x pertenece al fondo
            if self.fd(self.img[point[0],point[1]]):
                nothing_else = False
                lout.remove(point)
                lin.append(point)
                phi[point[0],point[1]] = -1
                #2.b Para todo vecino y de x, si matrix(y) = 3, agregar a LOUT y poner matrix(y) =1
                for aux_point in self.conexo4(point, width, height):
                    if phi[aux_point[0],aux_point[1]] == 3:
                        lout.append(aux_point)
                        phi[aux_point[0],aux_point[1]] = 1
        #3 Revisar los pixels en LIN que se transformaron en interiores y los borro de LIN y les pongo matrix(x) = -3
        new_interior = []
        for x in lin:
            # miro si cumple la definicion de lin
            belongs_lin = False
            for y in self.conexo4(x, width, height):
                if phi[y[0], y[1]] > 0:
                    belongs_lin = True
            if not belongs_lin:
                new_interior.append(x)
        for x in new_interior:
            lin.remove(x)
            phi[x[0],x[1]] = -3

        #4 Para cada LIN si Fd(x) < 0 borro de LIN y lo agrego a LOUT.
        for point in lin:
            if self.fd(self.img[point[0],point[1]]) == False:
                nothing_else = False
                lin.remove(point)
                lout.append(point)
                phi[point[0],point[1]] = 1
                #4.b Para todo vecino y de x con matrix(y) = -3, agregar a LIN y poner matrix(y) = -1
                for aux_point in self.conexo4(point, width, height):
                    if phi[aux_point[0],aux_point[1]] == -3:
                        lin.append(aux_point)
                        phi[aux_point[0],aux_point[1]] = -1
        #5 Revisar los pixels en LOUT que se transformaron en exterior y los borro de LOUT y les pongo matrix(x) = 3
        new_exterior = []
        for x in lout:
            # miro si cumple la definicion de lout
            belongs_lout = False
            for y in self.conexo4(x, width, height):
                if phi[y[0], y[1]] < 0:
                    belongs_lout = True
            if not belongs_lout:
                new_exterior.append(x)
        for x in new_exterior:
            lout.remove(x)
            phi[x[0], x[1]] = 3

        return [nothing_else, lin, lout, phi]

    #ToDo: cambiar por comentario para color
    def fd(self,pixel):
        # value = 1 - np.linalg.norm(pixel - self.object_avg) / 256 # (gray)
        p_obj = 1 - np.linalg.norm(pixel - self.object_avg) / (256**2 * 2)
        p_bg = 1 - np.linalg.norm(pixel - self.background_avg) / (256**2 * 2)
        value = np.log(p_obj/p_bg)
        return value > 0

    def conexo4(self, point, width, height):
        directions = [[1,0],[0,1],[-1,0],[0,-1]]
        neighbors = []
        for direc in directions:
            coordy = point[0] + direc[0]
            coordx = point[1] + direc[1]
            if coordx < 0 or coordy < 0 or coordx >= width or coordy >= height:
                break
            else:
                neighbors.append([int(coordy), int(coordx)])

        return neighbors

    # Flujo paso 2 (esto se llama con la seleccion ya hecha)
    def selection_prep(self, start, end):
        I = self.app_ref.get_processed()

        x_start, y_start = start
        x_end, y_end = end
        height, width = np.shape(I)[0:2]

        self.phi = np.zeros((height, width))
        self.img = np.copy(I)

        for y in range(height):
            for x in range(width):
                if y == y_start+1 or y == y_end-1:
                    if x > x_start and x < x_end:
                        self.phi[y,x] = -1
                        self.lin.append([y,x])
                    elif x == x_start or x == x_end:
                        self.phi[y,x] = 1
                        self.lout.append([y,x])
                    else:
                        self.phi[y,x] = 3
                elif y == y_start or y == y_end:
                    if x > x_start and x < x_end:
                        self.phi[y,x] = 1
                        self.lout.append([y,x])
                    else:
                        self.phi[y,x] = 3
                elif y > y_start and y < y_end:
                    if x == x_start+1 or x == x_end-1:
                        self.phi[y,x] = -1
                        self.lin.append([y,x])
                    elif x == x_start or x == x_end:
                        self.phi[y,x] = 1
                        self.lout.append([y,x])
                    elif x > x_start and x < x_end:
                        self.phi[y,x] = -3
                    else:
                        self.phi[y,x] = 3
                elif y < y_start or y > y_end:
                    self.phi[y,x] = 3

        # ToDo: cambiar por comentario para color
        self.update_avgs()
        # np.savetxt("phi.txt", self.phi, fmt="%s")
        contorno = self.contornos_activos_wrap(lin=self.lin, lout=self.lout, phi=self.phi, width=width,height=height)

        # Una vez que tengo los contornos, los marco en la imagen
        if utils.img_type(I) == 'GRAY': # si la imagen es de grises...
            # me armo una imagen rgb
            I = utils.join_bands(I,I,I)

        for y in range(height):
            for x in range(width):
                if contorno[y, x] == 1:
                    I[y, x] = (0, 0, 255) # azul para lout
                elif contorno[y, x] == -1:
                    I[y, x] = (255, 0, 0) # rojo para lin
                # alguno de los dos contornos se va a ver ;)

        self.app_ref.set_processed(I)

    def hough_lines(self):
        theta_step = askinteger("Hough", "Δθ (grados): ", initialvalue=5)
        rho_step = askinteger("Hough", "Δρ: ", initialvalue=10)
        epsilon = askfloat("Hough", "ε: ", initialvalue=1.2)
        threshold = askinteger("Hough", "Umbral: ", initialvalue=30)

        I = self.app_ref.get_processed()
        if utils.img_type(I) == 'RGB':
            messagebox.askokcancel('Error', 'La imagen debe ser binaria')

        img = hough_lines(np.array(I), theta_step=theta_step, rho_step=rho_step, epsilon=epsilon, threshold=threshold)

        self.app_ref.set_processed(Image.fromarray(img))

    def hough_circles(self):
        a_step = askinteger("Hough", "Δa: ", initialvalue=5)
        b_step = askinteger("Hough", "Δb: ", initialvalue=5)
        r_step = askinteger("Hough", "Δr: ", initialvalue=5)
        epsilon = askfloat("Hough", "ε: ", initialvalue=1.2)
        threshold = askinteger("Hough", "Umbral: ", initialvalue=30)

        I = self.app_ref.get_processed()
        if utils.img_type(I) == 'RGB':
            messagebox.askokcancel('Error', 'La imagen debe ser binaria')

        img = hough_circles(
            np.array(I),
            a_step=a_step, b_step=b_step, r_step=r_step,
            epsilon=epsilon,
            threshold=threshold
        )

        self.app_ref.set_processed(Image.fromarray(img))


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

def susan_count(img, row, col, threshold):
    mask = np.array([
        [0,0,1,1,1,0,0],
        [0,1,1,1,1,1,0],
        [1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1],
        [0,1,1,1,1,1,0],
        [0,0,1,1,1,0,0]
    ])

    count = 0
    height, width = np.shape(img)[0:2]
    center_value = int(img[row, col])

    for msk_row in range(7):
        for msk_col in range(7):
            if mask[msk_row, msk_col] == 0:
                continue

            img_row = row+msk_row-3
            img_col = col+msk_col-3

            if img_row < 0 or img_row >= height or img_col < 0 or img_col >= width:
                continue

            count += 1 if abs(center_value - int(img[img_row, img_col])) < threshold else 0

    return count


def susan(img, threshold=27, borders=True, corners=True):
    # paso img a una copia RGB para poder
    # marcar bordes y esquinas con colores
    img2 = utils.join_bands(
            np.array(img),
            np.array(img),
            np.array(img)
        )

    height, width = np.shape(img)

    for row in range(height):
        for col in range(width):
            s = 1.0 - (susan_count(img, row, col, threshold) / 37.0)
            if abs(s-0.5) < 0.15 and borders:
                img2[row, col] = (3, 152, 252)
            if abs(s-0.75) < 0.15 and corners:
                img2[row, col] = (252, 3, 107)

    return img2.astype('uint8')

def hough_lines(
  img,
  theta_step=15,
  rho_step=2,
  epsilon=1.0,
  threshold=50,
  max_lines=None
):
  '''
  Busca rectas en la imagen img (img debe ser binaria)
  '''

  h, w = np.shape(img)
  D = np.max([h,w])

  theta_count = int(np.pi/np.deg2rad(theta_step))
  rho_count = int(2*np.sqrt(2)*D/rho_step)
  print(theta_count, rho_count)

  # Funciones para pasar de indice a valor
  idx2theta = lambda idx: idx * np.deg2rad(theta_step) - np.pi/2
  idx2rho = lambda idx: (idx-rho_count/2) * rho_step

  # Creo la matriz acumulador teniendo en cuenta el rango
  # de tita y rho y la discretizacion de cada uno
  A = np.zeros((theta_count, rho_count))

  # Recorro los pixeles blancos y sumo
  # en los casos que cumplan la ecuacion de la recta
  for row in range(h):
    for col in range(w):
      # Solo miro pixeles blancos
      if img[row, col] < 255:
        continue

      for theta_idx in range(theta_count):
        theta = idx2theta(theta_idx)
        for rho_idx in range(rho_count):
          rho = idx2rho(rho_idx)

          # Veo si cumple la ecuacion de la recta
          if abs(rho - col*np.cos(theta) - row*np.sin(theta)) < epsilon:
            A[theta_idx, rho_idx] += 1

  # Busco la cantidad de votaciones del mas votado
  max_vot = np.max(A)
  print('La mas votada tiene {} votos'.format(max_vot))

  # Agrego las lineas a una priority queue ordenada por mas votaciones
  lines = []
  priority = lambda vot: max_vot - vot
  for theta_idx in range(theta_count):
    for rho_idx in range(rho_count):
      vots = A[theta_idx, rho_idx]
      if vots >= threshold:
        p = priority(A[theta_idx, rho_idx]) # prioridad de la linea
        l = (idx2theta(theta_idx), idx2rho(rho_idx)) # parametros  de la linea
        hpq.heappush(lines, (p, l))

  line_count = len(lines) if max_lines is None else np.min([len(lines), max_lines])
  print('Enconte {} lineas'.format(line_count))

  # Dibujo las rectas con mas puntaje
  img2 = np.zeros((h,w,3), dtype='uint8') # creo una imagen rgb para dibujar con color las rectas
  for row in range(h):
    for col in range(w):
      # Le asigno el color que tenia pero en RGB
      v = img[row,col]
      img2[row, col] = (v,v,v)
      # Veo si esta en alguna recta para pintarlo
      for l in range(line_count):
        theta, rho = lines[l][1]
        if abs(rho - col*np.cos(theta) - row*np.sin(theta)) < 0.5:
          # El pixel esta en la recta (lo pinto de rojo)
          img2[row, col] = (255,0,0)
          break; # Si ya lo pinte, no sigo viendo si hay mas lineas


  return img2

def hough_circles(
  img,
  a_step=2,
  b_step=2,
  r_step=50,
  epsilon=2,
  threshold=20,
  max_circles=None
):
  '''
  Busca circulos en la imagen img (img debe ser binaria)
  '''

  h, w = np.shape(img)
  D = np.max([h,w])

  a_count = int(w/a_step) + 1
  b_count = int(h/b_step) + 1
  r_count = int(D/r_step)

  # Creo la matriz acumulador
  A = np.zeros((a_count, b_count, r_count))

  # Recorro los pixeles blancos y sumo
  # en los casos que cumplan la ecuacion de la circunferencia
  for row in range(h):
    for col in range(w):
      # Solo miro pixeles blancos
      if img[row, col] < 255:
        continue

      for a_idx in range(a_count):
        a = a_idx*a_step
        for b_idx in range(b_count):
          b = b_idx * b_step
          for r_idx in range(r_count):
            r = (1+r_idx)*r_step
            # Veo si cumple la ecuacion de la circunferencia
            if abs(r**2 - (col-a)**2 - (row-b)**2) < epsilon:
              A[a_idx, b_idx, r_idx] += 1

  # Busco la cantidad de votaciones del mas votado
  max_vot = np.max(A)
  print('La mas votada tiene {} votos'.format(max_vot))

  # Agrego los circulos a una priority queue ordenada por mas votaciones
  circles = []
  priority = lambda vot: max_vot - vot
  for a_idx in range(a_count):
    a = a_idx*a_step
    for b_idx in range(b_count):
      b = b_idx * b_step
      for r_idx in range(r_count):
        r = (r_idx+1)*r_step
        vots = A[a_idx, b_idx, r_idx]
        if vots >= threshold:
          p = priority(vots) # prioridad del circulo
          circ = (a,b,r) # parametros  del circulo
          hpq.heappush(circles, (p, circ))


  circle_count = len(circles) if max_circles is None else np.min([len(circles),max_circles])
  print('Enconte {} circulos'.format(circle_count))

  for c in range(circle_count):
    print(circles[c])

  # Dibujo los circulos con mas puntaje
  img2 = np.zeros((h,w,3), dtype='uint8') # creo una imagen rgb para dibujar con color las rectas
  for row in range(h):
    for col in range(w):
      # Le asigno el color que tenia pero en RGB
      v = img[row,col]
      img2[row, col] = (v,v,v)
      # Veo si esta en alguna recta para pintarlo
      for l in range(circle_count):
        a, b, r = circles[l][1]
        if abs(r**2 - (col-a)**2 - (row-b)**2) < r:
          # El pixel esta en el circulo (lo pinto de rojo)
          img2[row, col] = (255,0,0)
          break; # Si ya lo pinte, no sigo viendo si esta en mas circulos

  return img2