import numpy as np

from tkinter.simpledialog import askinteger
from PIL import Image

class Gen():
    def __init__(self, app_ref):
        # Save app reference to
        # set the generated images
        self.app_ref = app_ref

    def inside_circle_rad(self, x, y, xc, yc, r):
        if ((x-xc)**2 + (y-yc)**2) <= r**2:
            return True
        else:
            return False

    def square(self):
        # Generate grayscale image
        img = Image.new('L', (200, 200))
        # Convert it to an array
        I = np.array(img)
        # Paint the corresponding pixels (a 100x100 square in the center)
        for x in range(49, 150):
            for y in range (49, 150):
                I[y][x] = 255

        # Convert the image back to a PIL binary image
        img = Image.fromarray(I).convert('1')
        
        # Set the image as the original (and processed)
        self.app_ref.set_original(img)

    def circle(self):
        # Ask integer for r, max 200
        r = askinteger("Circle", "Radius: ",
                  initialvalue=50,
                  minvalue=2,
                  maxvalue=100)
        #Generate grayscale image
        img = Image.new('L', (200,200))
        # to array
        I = np.array(img)
        # Paint pixels (center 100x100 and radius r)
        for x in range(0, 200):
            for y in range(0, 200):
                if(self.inside_circle_rad(x, y, 100, 100, r)):
                    I[y][x] = 255

        # Convert the image back to a PIL binary image
        img = Image.fromarray(I).convert('1')
        
        # Set the image as the original (and processed)
        self.app_ref.set_original(img)

    # Converts an x coordinte to the
    # corresponding gray color
    def position_to_gray(self, x, img_width):
        return int(x * (255.0/img_width))

    def gray_gradient(self):
        img_width = 200
        # Generate grayscale image
        img = Image.new('L', (img_width, img_width))
        # to array
        I = np.array(img)
        # Paint pixels using a linear horizontal gradient
        for x in range(0, img_width):
            for y in range(0, img_width):
                I[y][x] = self.position_to_gray(x, img_width)

        # convert back to PIL
        img = Image.fromarray(I)

        # Set as the original image
        self.app_ref.set_original(img)
