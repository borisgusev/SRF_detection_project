import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import util, exposure, feature
from tqdm import tqdm

import get_img_paths
import retina_mask
import image_preprocessing
import segmentation


def find_dark_blobs(img):
    """Uses Laplacian of Gaussians to identify round regions (blobs) that are darker that their surroundsings

    These blobs are candidates for sub-retinal fluid which typically is seens as a dark 'blob' within the retina.

    Args:
        img (Image Array): grayscale image

    Returns:
        2D Array: Each row is y-coord, x-coord, radius
    """
    # invert image as exposure.blob_log finds light blobs, whereas SRF is dark
    img = util.invert(img)
    # gamma exposure seems to increase sensitivity. another parameter to tinker with
    img = exposure.adjust_gamma(img, gamma=2.5)
    blobs = feature.blob_log(img,
                             min_sigma=1,
                             max_sigma=20,
                             num_sigma=20,
                             threshold=0.20,
                             overlap=1,
                             exclude_border=(65))
    # convert sigma vals in third to column to radii
    blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
    return blobs


def filter_blobs(img, blobs):
    """Filters candidate blobs to eliminate non-SRF blobs

    Args:
        img (Image Array): grayscale image
        blobs (2D Array): each row is y-coord, x-coord, radius

    Returns:
        2D Array: filtered blobs
    """
    # first filter: dark regions of retina
    seg_img, labels = segmentation.segmentation(img, nclust=6)
    fluid = labels == 1
    ys, xs = blobs[:, 0].astype('int64'), blobs[:, 1].astype('int64')
    blobs = blobs[np.nonzero(fluid[ys, xs])]
    # second filter: not too far above RPE layer
    rpe_edge = retina_mask.rpe_upper_edge(img)
    thresh = 5
    bool_mask = np.zeros(blobs.shape[0], dtype='bool')
    for i, blob in enumerate(blobs):
        y, x, r = blob.astype('int64')
        if np.any(rpe_edge[y:y + r + thresh, x]):
            bool_mask[i] = True
    blobs = blobs[bool_mask]
    return blobs


def plot_blobs(axes, blobs):
    """Utility function to plot blobs in-place on axes

    Args:
        axes: single set of matplotlib axes
        blobs (2D Array): each row is y-coord, x-coord, radius
    """
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
        axes.add_patch(c)


def plot_blobbing_process(img):
    fig, axes = plt.subplots(nrows=2, ncols=2)
    ax = axes.ravel()
    list(map(lambda x: x.set_axis_off(), ax))

    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Original')

    seg_img, labels = segmentation.segmentation(img, nclust=6)
    ax[1].imshow(labels == 1, cmap='gray')
    ax[1].set_title('Segment Mask')

    ax[2].imshow(retina_mask.rpe_upper_edge(img), cmap='gray')
    ax[2].set_title('Upper RPE-layer edge estimate')

    ax[3].imshow(img, cmap='gray')
    blobs = find_dark_blobs(img)
    blobs = filter_blobs(img, blobs)
    plot_blobs(ax[3], blobs)
    ax[3].set_title('Blobs')  # change title?
    return fig, ax


if __name__ == '__main__':
    # Takes a little while to run, generates folders
    # with figures for each healthy and srf image

    output_path = Path('blob_output')
    output_path.mkdir(exist_ok=True)
    healthy, srf = get_img_paths.train_data()

    # img = plt.imread(srf[3])
    # img = image_preprocessing.preprocess(img)
    # fig, axes = plot_before_after(img)
    # plt.tight_layout()
    # plt.show()

    healthy_output_path = output_path / 'NoSRF'
    healthy_output_path.mkdir(exist_ok=True)
    for img_path in tqdm(healthy):
        img = plt.imread(img_path)
        img = image_preprocessing.preprocess(img)
        fig, axes = plot_blobbing_process(img)
        fig.suptitle(img_path.name)
        plt.tight_layout()
        file_name = healthy_output_path / img_path.name
        plt.savefig(file_name, dpi=400, bbox_inches='tight')
        # plt.show()
        plt.close()

    srf_output_path = output_path / 'SRF'
    srf_output_path.mkdir(exist_ok=True)
    for img_path in tqdm(srf):
        img = plt.imread(img_path)
        img = image_preprocessing.preprocess(img)
        fig, axes = plot_blobbing_process(img)
        fig.suptitle(img_path.name)
        plt.tight_layout()
        file_name = srf_output_path / img_path.name
        plt.savefig(file_name, dpi=400, bbox_inches='tight')
        # plt.show()
        plt.close()
