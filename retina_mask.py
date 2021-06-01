from skimage import filters
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
import scipy.signal
from tqdm import tqdm

import image_preprocessing
import get_img_paths
import segmentation
from pathlib import Path


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
    # Script to apply masking to image

    output_path = Path('out_retina_mask')
    output_path.mkdir(exist_ok=True)

    test_img = get_img_paths.get_test_data()

    for img_name in tqdm(test_img):

        # Original image
        img = plt.imread(img_name)
        
        plt.subplot(1, 3, 1)
        plt.title ("Original", fontsize=10)
        plt.imshow(img, cmap='gray')
        plt.axis("off")

        # Processed image
        img = image_preprocessing.preprocess(img)

        plt.subplot(1, 3, 2)
        plt.title ("Filtered", fontsize=10)
        plt.imshow(img, cmap='gray')
        plt.axis("off")

        # Mask. Uncomment one of interest.
        mask = rpe_upper_edge(img)
        # mask = retinal_mask(img)
        
        plt.subplot(1, 3, 3)
        plt.title ("Mask", fontsize=10)
        plt.imshow(mask, cmap='gray')
        plt.axis("off")

        file_name = output_path / img_name.name
        plt.savefig(file_name, dpi=400, bbox_inches='tight')
        # plt.show()
        plt.close()


