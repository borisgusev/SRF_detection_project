from skimage import filters
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
import scipy.signal

import image_preprocessing
import get_img_paths
import segmentation


def retinal_mask(img):
    """Generates a mask of the retina using thresholding

    Args:
        img (Image Array): Grayscale Image

    Returns:
        2D Array: Binary Mask
    """
    mask = np.zeros_like(img)
    blur = filters.gaussian(img, sigma=10)
    mask[blur > 0.25] = 1
    mask = ndi.binary_fill_holes(mask)
    return mask


def rpe_upper_edge(img):
    """Generates a mask for estimate of the upper edge of RPE layer

    Args:
        img (Image Array): Grayscale Image

    Returns:
        2D Array: Binary Mask
    """
    # find brightest segment
    nclust = 4
    seg_img, labels = segmentation.segmentation(img, nclust=nclust)
    mask = labels == nclust - 1
    # find relevant edges of mask, and only keep bottom-most edge pixels
    mask = filters.sobel_h(mask) > 0
    for col in range(mask.shape[1]):
        indices = np.nonzero(mask[:, col])[0]
        mask[indices[:-1], col] = 0
    # fill in gaps
    ys, xs = np.nonzero(mask)
    sorted_indices = np.argsort(xs)
    new_xs = np.arange(np.min(xs), np.max(xs) + 1)
    new_ys = np.interp(new_xs, xs[sorted_indices], ys[sorted_indices])
    # smooth curve
    new_ys = scipy.signal.medfilt(new_ys, kernel_size=31)
    new_ys = new_ys.astype('int64')
    # create new mask
    mask = np.zeros_like(img)
    mask[new_ys, new_xs] = 1
    return mask


if __name__ == '__main__':
    # Script to apply masking to image
    healthy, srf = get_img_paths.train_data()
    img = plt.imread(srf[0])
    img = image_preprocessing.preprocess(img)
    # Uncomment one of interest
    mask = rpe_upper_edge(img)
    # mask = retinal_mask(img)

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')

    plt.show()
