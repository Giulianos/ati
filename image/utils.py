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

    return np.uint8((I-np.min(I))/np.ptp(I)*max_value)

def img_type(I):
    # len 3: row, col & channel
    if len(np.shape(I)) == 3:
        return 'RGB'
    
    # len 2: row, col
    if len(np.shape(I)) == 2:
        if I.dtype == bool:
            return 'BIN'
        else:
            return 'GRAY'
    
    # unknown type
    return None

def split_bands(I):
    rows, cols, bands = np.shape(I)
    R = np.zeros((rows, cols))
    G = np.zeros((rows, cols))
    B = np.zeros((rows, cols))

    for row in range(rows):
        for col in range(cols):
            r, g, b = I[row][col]
            R[row][col] = r
            G[row][col] = g
            B[row][col] = b
    
    return R, G, B

def join_bands(R, G, B):
    if np.shape(R) != np.shape(G) or np.shape(R) != np.shape(B):
        return None
    rows, cols = np.shape(R)
    I = np.zeros((rows, cols, 3))
    for row in range(rows):
        for col in range(cols):
            I[row][col][0] = R[row][col]
            I[row][col][1] = G[row][col]
            I[row][col][2] = B[row][col]
    
    return I

def apply_gray_to_rgb(I, func):
    R, G, B = split_bands(I)
    I2 = join_bands(
        func(R),
        func(G),
        func(B)
    )

    return I2

#rotate a 3x3 matrix times *45deg clockwise
def rotate_matrix3(m,times):
    m2 = np.array(m)
    #ordered elements to shift indexes
    idx_order = [(0,0), (0,1), (0,2), (1,2), (2,2), (2,1), (2,0), (1,0)]
    for i in range(len(idx_order)):
        # this works because negative indices
        # start from the en (index - 1 is the last element)
        m2[idx_order[i]] = m[idx_order[i-times]]

    return m2