import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import feature 
from skimage import color
from skimage import filters
from skimage import util
from skimage import segmentation
from skimage import exposure

# assumes that the unzipped training folder is included in the directory
train_data = Path('Train-Data')
healthy_images = [img_path for img_path in (train_data / 'NoSRF').iterdir()]
srf_images = [img_path for img_path in (train_data / 'SRF').iterdir()]

# placeholder to view individual images
# img = plt.imread(healthy_images[5])
for i in healthy_images:
# for i in srf_images:
    img = plt.imread(i)
    img = color.rgba2rgb(img)
    x = color.rgb2gray(img)
    # x = exposure.equalize_hist(x)
    x = util.invert(x)
    x = exposure.adjust_gamma(x, gamma = 2.5)
    # edge = filters.sobel(filters.gaussian(img, sigma=5))
    # edge = util.invert(edge)
    # edge = feature.canny(edge) 
    # edge = feature.canny(img, sigma = 4)
    # edge = filters.hessian(img)
    # edge = filters.sato(img)

    # processed = exposure.equalize_hist(img)
    blobs = feature.blob_log(x, min_sigma=3, max_sigma=15, num_sigma=20, threshold=0.18, exclude_border=(65))
    blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
    # coords = blobs[:, :2].astype('int64')
    # processed[coords[:, 0], coords[:, 1]] = (1,0,0)

    # plt.subplot(1, 2, 1)
    # plt.imshow(img, cmap='gray')
    # plt.subplot(1, 2, 2)
    # plt.imshow(processed)
    # plt.show()

    # fig, axes = filters.try_all_threshold(img)
    # plt.show()

    fig, axes = plt.subplots(1, 2,  sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(img)
    ax[1].imshow(img)
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
        ax[1].add_patch(c)
    ax[0].set_axis_off()
    ax[1].set_axis_off()

    plt.tight_layout()
    plt.show()