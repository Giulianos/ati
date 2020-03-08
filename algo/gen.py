import numpy as np

from PIL import Image

class Gen():
    def __init__(self, app_ref):
        # Save app reference to
        # set the generated images
        self.app_ref = app_ref

    def square(self):
        # Generate grayscale image
        img = Image.new('L', (200, 200))
        # Convert it to an array
        I = np.array(img)
        # Paint the corresponding pixels (a 100x100 square in the center)
        for x in range(49, 150):
            for y in range (49, 150):
                I[x][y] = 255
        # Convert the image back to a PIL binary image
        img = Image.fromarray(I).convert('1')
        
        # Set the image as the original (and processed)
        self.app_ref.set_original(img)

