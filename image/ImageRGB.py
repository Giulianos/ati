from image.Image import Image

import numpy as np

class ImageRGB(Image):
  def __init__(self, src=None, width=None, height=None):
    # To do: handle cases were src is != None
    if src == None:
      self.array = np.zeros((height, width, 3))
  
    self.height, self.width = np.shape(self.array)[0:2]

  def shape(self):
    return self.height, self.width, 3