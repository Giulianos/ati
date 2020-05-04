from image.Image import Image
from image.ImageRGB import ImageRGB

import numpy as np

def square():
  img = Image(width=200, height=200)
  # Paint the corresponding pixels (a 100x100 square in the center)
  for x in range(49, 150):
    for y in range (49, 150):
      img[y, x] = 255
  
  return img

def circle(r=50):
  img = Image(width=200, height=200)

  # Paint pixels (center 100x100 and radius r)
  for x in range(0, 200):
    for y in range(0, 200):
      if inside_circle_rad(x, y, 100, 100, r):
        img[y,x] = 255

  return img

def gray_gradient():
  img_width = 200
  img = Image(width=img_width, height=img_width)
  # Paint pixels using a linear horizontal gradient
  for x in range(0, img_width):
      for y in range(0, img_width):
          img[y,x] = position_to_gray(x, img_width)
  
  return img

def color_gradient():
  img_width = 200
  img = ImageRGB(width=img_width, height=img_width)

  # Paint pixels using a linear horizontal gradient
  for x in range(0, img_width):
    for y in range(0, img_width):
      img[y,x] = position_to_rgb(x, img_width)

  return img


# helper functions
def inside_circle_rad(x, y, xc, yc, r):
  if ((x-xc)**2 + (y-yc)**2) <= r**2:
    return True
  else:
    return False

# Converts an x coordinte to the
# corresponding gray color
def position_to_gray(x, img_width):
  return int(x * (255.0/img_width))

# converts an x coordinate to the
# corresponding rgb tuple
def position_to_rgb(x, img_width):
  # Generate colors using cos:
  # https://www.desmos.com/calculator/lakvddkg1g
  r = np.cos((1.0*x/img_width)*np.pi)*255
  g = np.cos((1.0*x/img_width)*np.pi - 0.5*np.pi)*255
  b = np.cos((1.0*x/img_width)*np.pi - np.pi)*255

  r = 0 if r < 0 else r
  g = 0 if g < 0 else g
  b = 0 if b < 0 else b

  return (r,g,b)