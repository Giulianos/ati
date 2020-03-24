import numpy as np

from PIL import Image

from tkinter.simpledialog import askfloat, askinteger

from algo.utils import calculate_histogram

class Functions():
    def __init__(self, app_ref):
        self.app_ref = app_ref

    def negative(self):
        img = self.app_ref.img_proc
        bands = img.getbands()
        if bands == ('1',):
            img = img.convert('L')

        I = np.array(img)

        for pixel in np.nditer(I, op_flags=['readwrite']):
            if bands == ('1',):
                pixel[...] = 0 if pixel else 255
            else:
                pixel[...] = np.subtract(255, pixel)

        img = Image.fromarray(I)
        if bands == ('1',):
            img = img.convert('1')
        self.app_ref.set_processed(img)
    
    def multiply_by_scalar(self):
        scalar = askfloat("Multiplicación por escalar", "Escalar: ",
                  initialvalue=1,
                  minvalue=0,
                  maxvalue=255)

        img = self.app_ref.img_proc
        bands = img.getbands()
        if bands == ('1',):
            img = img.convert('L')

        I = np.array(img)

        for pixel in np.nditer(I, op_flags=['readwrite']):
            if bands == ('1',):
                pixel[...] = np.multiply(scalar, pixel)

        img = Image.fromarray(I)
        if bands == ('1',):
            img = img.convert('1')
        self.app_ref.set_processed(img)

    def sum_other_image(self):
        self.apply_binary_op(lambda p1,p2: p1+p2)

    def substract_other_image(self):
        self.apply_binary_op(lambda p1,p2: p1-p2)

    def multiply_by_other_image(self):
        self.apply_binary_op(lambda p1,p2: p1*p2)

    def apply_binary_op(self, op):
        # load other image
        img_other = self.app_ref.load_image_from_file()

        if img_other.getbands() == ('R','G','B') or self.app_ref.img_proc.getbands() == ('R','G', 'B'):
            # TODO: Not implemented for RGB
            return

        # TODO: Handle different size images

        # Convert images to arrays
        I1 = np.array(self.app_ref.img_proc.convert('L'), dtype=float)
        I2 = np.array(img_other.convert('L'), dtype=float)

        # Perform function pixel by pixel
        for pix1, pix2 in np.nditer([I1, I2], op_flags=['readwrite']):
            pix1[...] = op(pix1, pix2)


        # Remap image to 0-255
        img = Image.fromarray(self.remap_image_array(I1))

        self.app_ref.set_processed(img)


    # This function maps arbitrary
    # pixel values to 0-max_value
    def remap_image_array(self, I, max_value=255):
        # Handle images with range 0
        if np.ptp(I) == 0:
            if np.min(I) < 0:
                return I-np.min(I)
            elif np.max(I) > max_value:
                return I-(np.max(I)-max_value)
            else:
                return I

        return (I-np.min(I))/np.ptp(I)*max_value


    def equalize_histogram(self):
        # Image to array
        I = np.array(self.app_ref.img_proc.convert('L'))

        # Calculate histogram for image
        hist = calculate_histogram(I, False)

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

        # Set equalized image
        img = Image.fromarray(I)
        self.app_ref.set_processed(img)

    def gen_gauss(self, mu, desvio):
        return np.random.normal(mu, desvio)

    def gen_raleigh(self, xhi):
        return np.random.rayleigh(xhi)

    def gen_exp(self, lamb):
        return np.random.exponential(1/lamb)
    
    def gen_uniform(self, min, max):
        return np.random.uniform(min, max)

    def noise_additive_gauss(self):
        percentage = askinteger("Exponential Noise", "Porcentaje a contaminar: ",
                    initialvalue=30)
        mu = askfloat("Distribucion Gaussiana", "Variable μ: ",
                  initialvalue=1)
        desvio = askfloat("Distribucion Gaussiana", "Variable σ: ",
                  initialvalue=1)

        self.apply_noise(percentage, "gauss", "add", mu, desvio)
        
        print("Additive Gauss applied!")
        return 0
    
    def noise_multiplicative_raleigh(self):
        percentage = askinteger("Exponential Noise", "Porcentaje a contaminar: ",
                    initialvalue=30)
        xhi = askfloat("Distribucion Raleigh", "Variable ξ: ",
                  initialvalue=1)
        
        self.apply_noise(percentage, "raleigh", "mul", xhi, None)

        print("Multiplicative Raleigh applied!")
        return 0
    
    def noise_multiplicative_exp(self):
        percentage = askinteger("Exponential Noise", "Porcentaje a contaminar: ",
                    initialvalue=30)
        lamb = askfloat("Distribucion Exponencial", "Variable λ: ",
                  initialvalue=1)

        self.apply_noise(percentage, "exp", "mul", lamb, None)

        print("Multiplicative exponential applied!")
        return 0

    def apply_noise(self, percentage, op_operation, op_type, var1, var2):
        #iter over image
        img = self.app_ref.img_proc
        bands = img.getbands()
        if bands == ('1',):
            img = img.convert('L')

        I = np.array(img)

        for pixel in np.nditer(I, op_flags=['readwrite']):
            noised = self.gen_uniform(0,100)
            if noised <= percentage:
                if op_operation == "exp":
                    random_number = self.gen_exp(var1)
                elif op_operation == "raleigh":
                    random_number = self.gen_raleigh(var1)
                elif op_operation == "gauss":
                    random_number = self.gen_gauss(var1, var2)
                
                if op_type == "mul":
                    pixel[...] = pixel*random_number
                elif op_type == "add":
                    pixel[...] = pixel+random_number
            
            #ToDo for RGB

        img = Image.fromarray(I)
        if bands == ('1',):
            img = img.convert('1')
        self.app_ref.set_processed(img)



