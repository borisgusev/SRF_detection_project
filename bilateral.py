import numpy as np
import matplotlib.pyplot as plt
from skimage import color, feature, filters, segmentation, morphology, measure
import time

def filter_bilateral( img_in, sigma_s, sigma_v, reg_constant=1e-8 ):
    """Bilateral filtering of an input image """

    # check the input
    if not isinstance( img_in, np.ndarray ) or img_in.dtype != 'float32' or img_in.ndim != 2:
        raise ValueError('Bilateral filter: expected a grayscale image!')

    # Gaussian function 
    gaussian = lambda r2, sigma: (np.exp( -0.5*r2/sigma**2 )*3).astype(int)*1.0/3.0

    # define the window width to be the 3 time the spatial std. dev. to 
    # be sure that most of the spatial kernel is actually captured
    win_width = int( 3*sigma_s+1 )

    # initialize the results and sum of weights to very small values for
    # numerical stability. not strictly necessary but helpful to avoid
    # wild values with pathological choices of parameters
    wgt_sum = np.ones( img_in.shape )*reg_constant
    result  = img_in*reg_constant

    # accumulate the result by circularly shifting the image across the
    # window in the horizontal and vertical directions. within the inner
    # loop, calculate the two weights and accumulate the weight sum and 
    # the unnormalized result image

    for shft_x in range(-win_width,win_width+1):
        for shft_y in range(-win_width,win_width+1):
            # compute the spatial weight
            w = gaussian( shft_x**2+shft_y**2, sigma_s )
            # shift by the offsets
            off = np.roll(img_in, [shft_y, shft_x], axis=[0,1] )
            # compute the value weight
            weight = w*gaussian((off-img_in)**2, sigma_v)
            result += off*weight
            wgt_sum += weight

    # normalize the result and return
    return result/wgt_sum
