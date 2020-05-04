import numpy as np
from PIL import Image as PILImage

import image.utils as utils

class Image:

  # the image will be created depending on the type of src:
  # - None: width and height must be passed (the created image is black)
  # - String: the filename of the image (if its RAW, width and height must be passed)
  # - Numpy array or PIL Image: the image data
  def __init__(self, src=None, width=None, height=None):
    # Open the image from the file
    if type(src) == str:
      if ".RAW" in src or ".raw" in src:
        with open(src, "rb") as binary_file:
          databytes = binary_file.read()
          self.pil = PILImage.frombytes("L", (width, height), databytes, decoder_name='raw')
      else: 
        self.pil = PILImage.open(src)
      self.array = np.array(self.pil)
    elif type(src) == PILImage:
      self.pil = src.copy()
      self.array = np.array(self.pil)
    elif src == None:
      self.array = np.zeros((height, width))
      self.pil = self.get_PIL()
    else:
      self.array = np.array(src)
      self.pil = self.get_PIL()
    
    # Init properties
    self.height, self.width = np.shape(self.array)

  # Indexing is passed to the underlying np array
  def __getitem__(self, key):
    y,x = key
    if x < 0 or x >= self.width or y < 0 or y >= self.height:
      raise IndexError
    return self.array[key]
  
  def __setitem__(self, key, value):
    y,x = key
    if x < 0 or x >= self.width or y < 0 or y >= self.height:
      raise IndexError
    self.array[key] = value
  
  def get_width(self):
    return self.width
  
  def get_height(self):
    return self.height
  
  def shape(self):
    return self.height, self.width, 1
  
  def get_PIL(self):
    return PILImage.fromarray(utils.remap_image(self.array))
  
  def get_array(self):
    return self.array

  def save(self, filename):
    self.get_PIL().save(filename)

  def show(self):
    self.get_PIL().show()