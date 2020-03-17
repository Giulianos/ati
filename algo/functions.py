import numpy as np

from PIL import Image

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

    def sum_other_image(self):
        # load other image
        img_other = self.app_ref.load_image_from_file()

        # Check if both images have the same bands
        if img_other.getbands() != self.app_ref.img_proc.getbands():
            # TODO: Show error and do nothing
            return
        elif img_other.getbands() == ('R','G','B'):
            # TODO: Not implemented for RGB
            return

        # TODO: Handle different size images

        # Convert images to arrays
        I1 = np.array(self.app_ref.img_proc.convert('L'))
        I2 = np.array(img_other.convert('L'))

        for pix1, pix2 in np.nditer([I1, I2], op_flags=['readwrite']):
            pix1[...] += pix2

        # Remap image to 0-255
        img = Image.fromarray(self.remap_image_array(I1))

        if img_other.getbands() == ('1',):
            img = img.convert('1')

        self.app_ref.set_processed(img)


    # This function maps arbitrary
    # pixel values to 0-255
    def remap_image_array(self, I):
        I2 = 255 * (I - np.min(I))/np.ptp(I)

        return I2
