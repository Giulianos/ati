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

