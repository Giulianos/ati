import numpy as np

# Calculate histogram for grayscale images
def calculate_histogram(img_array, relative=True):
    bins = np.zeros(256)
    pixel_count = 0

    for pixel in np.nditer(img_array):
        bins[pixel] += 1
        pixel_count += 1

    if relative:
        # Convert to relative frequencies
        bins /= pixel_count

    return bins

# This function maps arbitrary
# pixel values to 0-max_value
def remap_image(I, max_value=255):
    # Handle images with range 0
    if np.ptp(I) == 0:
        if np.min(I) < 0:
            return I-np.min(I)
        elif np.max(I) > max_value:
            return I-(np.max(I)-max_value)
        else:
            return I

    return (I-np.min(I))/np.ptp(I)*max_value

