import numpy as np
from scipy.ndimage import morphology
# import matplotlib.pyplot as plt
from skimage import color, restoration, exposure, filters


def rm_white_frame(img):
    '''removes white frame around image'''

    # create a negative (to invert the white frame into black, to fill with 0)
    img = 1 - img

    h, w, d = img.shape
    #left limit
    for i in range(w):
        if np.sum(img[:, i, :]) > 0:
            break
    #right limit
    for j in range(w - 1, 0, -1):
        if np.sum(img[:, j, :]) > 0:
            break

    #top limit
    for k in range(h):
        if np.sum(img[k, :, :]) > 0:
            break
    #bottom limit
    for l in range(h - 1, 0, -1):
        if np.sum(img[l, :, :]) > 0:
            break

    cropped = img[k:l + 1, i:j + 1, :].copy()
    #back to normal
    cropped = 1 - cropped

    return cropped


def preprocess(img):
    img = color.rgba2rgb(img)
    img = rm_white_frame(img)
    img = color.rgb2gray(img)

    img = restoration.denoise_nl_means(img)
    # img = restoration.denoise_bilateral(img)
    # img = exposure.equalize_adapthist(img)

    img = filters.gaussian(img, sigma = 3)
    img = filters.unsharp_mask(img, radius = 20, amount = 2)

    return img