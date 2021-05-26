from skimage import color, feature, filters, segmentation, morphology, measure
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

import image_preprocessing
import get_file_paths
import find_blobs
import image_edit_utils as utls


def retinal_mask(img):
    mask = np.zeros_like(img)
    blur = filters.gaussian(img, sigma=10)
    mask[blur > 0.25] = 1
    mask = ndi.binary_fill_holes(mask)
    return mask


if __name__ == '__main__':
    healthy, srf = get_file_paths.get_all_train_data()
    img = plt.imread(srf[4])
    # for img in srf:
    # img = plt.imread(img)
    img = image_preprocessing.preprocess(img)

    # img = morphology.opening(img)

    # edges = get_retinal_mask(img)
    edges = retinal_mask(img)

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap='gray')

    plt.show()
