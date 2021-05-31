from numpy.core.fromnumeric import argmax
from skimage import color, feature, filters, segmentation, morphology, measure
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
import scipy.signal

import image_preprocessing
import get_file_paths
import find_blobs
import image_edit_utils as utls


def retinal_mask(img):
    mask = np.zeros_like(img)
    blur = filters.gaussian(img, sigma=10)
    mask[blur > 0.2] = 1
    mask = ndi.binary_fill_holes(mask)
    return mask


def rpe_upper_edge(img):
    # mask = retinal_mask(img)
    # mask = filters.sobel_h(mask) < 0
    # for col in range(mask.shape[1]):
    #     indices = np.nonzero(mask[:, col])[0]
    #     mask[indices[:-1], col] = 0
    # img = filters.sobel(img)
    mask = np.zeros_like(img)

    sorted_indices = np.argsort(img, axis=None)

    # find the top x% brightest pixels
    top_percentile = 0.05
    top_number = int(sorted_indices.size * top_percentile)
    # convert flat array indices into shaped-array indices
    top_indices = sorted_indices[-top_number:]
    top_indices = np.unravel_index(top_indices, shape=img.shape)
    mask[top_indices] = 1

    # for each column, from the bottom, take the first pixel where tha mask changes from 1 to 0

    mask = filters.sobel_h(mask) > 0
    # plt.subplot(1, 3, 2)
    # plt.imshow(mask, cmap='gray')

    prev = -1
    for col in range(0,mask.shape[1]):
        indices = np.nonzero(mask[:, col])[0]

        if (len(indices)>0):
            max_ind = indices[len(indices)-1]
            if (np.abs(max_ind - prev) > 20 and prev != -1):
                mask[indices, col] = 0
            else:
                prev = max_ind
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
    healthy, srf, test_images = get_file_paths.get_all_train_data()
    #img = plt.imread(srf[5])
    for img in test_images:
        img = plt.imread(img)

        plt.subplot(1, 3, 1)
        plt.imshow(img, cmap='gray')

        img = image_preprocessing.preprocess(img)

        # img = morphology.opening(img)

        # edges = get_retinal_mask(img)
        # mask = retinal_mask(img)
        mask = rpe_upper_edge(img)

        plt.subplot(1, 3, 3)
        plt.imshow(mask, cmap='gray')

        plt.show()
        plt.close()
