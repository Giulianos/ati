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
        # Create image array
        I = np.zeros((200, 200), dtype='int64')
        # Paint the corresponding pixels (a 100x100 square in the center)
        for x in range(49, 150):
            for y in range (49, 150):
                I[y][x] = 255

        # Set the image as the original (and processed)
        self.app_ref.set_original(I)

    def circle(self):
        # Ask integer for r, max 200
        r = askinteger("Circle", "Radius: ",
                  initialvalue=50,
                  minvalue=2,
                  maxvalue=100)
        if type(r) != int:
            r = 50

        # Create image array
        I = np.zeros((200,200), dtype='int64')
        # Paint pixels (center 100x100 and radius r)
        for x in range(0, 200):
            for y in range(0, 200):
                if(self.inside_circle_rad(x, y, 100, 100, r)):
                    I[y][x] = 255

        # Set the image as the original (and processed)
        self.app_ref.set_original(I)

    # Converts an x coordinte to the
    # corresponding gray color
    def position_to_gray(self, x, img_width):
        return int(x * (255.0/img_width))

    def gray_gradient(self):
        img_width = 200
        # Create image array
        I = np.zeros((img_width, img_width), dtype='int64')
        # Paint pixels using a linear horizontal gradient
        for x in range(0, img_width):
            for y in range(0, img_width):
                I[y][x] = self.position_to_gray(x, img_width)

        # Set as the original image
        self.app_ref.set_original(I)

    # converts an x coordinate to the
    # corresponding rgb tuple
    def position_to_rgb(self, x, img_width):
        # Generate colors using cos:
        # https://www.desmos.com/calculator/lakvddkg1g
        r = np.cos((1.0*x/img_width)*np.pi)*255
        g = np.cos((1.0*x/img_width)*np.pi - 0.5*np.pi)*255
        b = np.cos((1.0*x/img_width)*np.pi - np.pi)*255

        r = 0 if r < 0 else r
        g = 0 if g < 0 else g
        b = 0 if b < 0 else b

        return (r,g,b)

    def color_gradient(self):
        img_width = 200
        # Create image array
        I = np.zeros((img_width, img_width, 3), dtype='int64')

        # Paint pixels using a linear horizontal gradient
        for x in range(0, img_width):
            for y in range(0, img_width):
                I[y][x] = self.position_to_rgb(x, img_width)

        # Set as the original image
        self.app_ref.set_original(I)


