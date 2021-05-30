import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# from scipy.ndimage import morphology
from skimage import color, util, exposure, feature, morphology, filters
from tqdm import tqdm
from scipy import ndimage as ndi

import get_file_paths
import retina_mask
import image_preprocessing
import segmentation


def find_candidate_srf_blobs(img):
    # invert image as exposure.blob_log finds light blobs, whereas SRF is dark
    img = util.invert(img)
    # gamma exposure seems to increase sensitivity. another parameter to tinker with
    img = exposure.adjust_gamma(img, gamma=2.5)
    blobs = feature.blob_log(img,
                             min_sigma=1,
                             max_sigma=20,
                             num_sigma=20,
                             threshold=0.12,
                             overlap=1,
                             exclude_border=(65))
    # convert sigma vals in third to column to radii
    blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
    return blobs


def filter_blob_candidates(img, blobs):
    # mask = retina_mask.retinal_mask(img)
    # mask = morphology.binary_erosion(mask, selem=morphology.rectangle(25, 1))
    # y, x = blobs[:, 0].astype('int64'), blobs[:, 1].astype('int64')
    # blobs = blobs[np.where(mask[y, x])]
    
    # blurred = filters.gaussian(img, sigma = 1)
    seg_img, labels = segmentation.segmentation(img, nclust=6)
    sorted_labels = segmentation.sort_labels(seg_img, labels)
    fluid = labels == sorted_labels[1]
    # fluid = ndi.binary_fill_holes(fluid)
    # fluid = np.logical_not(fluid)
    ys, xs = blobs[:, 0].astype('int64'), blobs[:, 1].astype('int64')
    blobs = blobs[np.nonzero(fluid[ys, xs])]


    rpe_edge = retina_mask.rpe_upper_edge(img)
    thresh = 15
    bool_mask = np.zeros(blobs.shape[0], dtype='bool')
    for i, blob in enumerate(blobs):
        y, x, r = blob.astype('int64')
        if np.any(rpe_edge[y : y+r+thresh, x]):
            bool_mask[i] = True
    blobs = blobs[bool_mask]
    return blobs


def plot_blobs(axes, blobs):
    # plot each blob as circle at the coordinate with respective radius
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
        axes.add_patch(c)
    # return axes


def plot_before_after(img):
    # utility function to plot original and blobs side by side
    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Original')
    ax[2].imshow(img, cmap='gray')
    ax[2].set_title('Candidate Blobs')
    blobs = find_candidate_srf_blobs(img)
    blobs = filter_blob_candidates(img, blobs)
    plot_blobs(ax[2], blobs)
    ax[0].set_axis_off()
    ax[2].set_axis_off()
    return fig, ax


if __name__ == '__main__':
    # Takes a little while to run, generates folders
    # with figures for each healthy and srf image

    output_path = Path('blob_output')
    output_path.mkdir(exist_ok=True)
    healthy, srf = get_file_paths.get_all_train_data()

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
        fig, axes = plot_before_after(img)
        axes[1].set_axis_off()
        axes[1].imshow(retina_mask.rpe_upper_edge(img))
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
        fig, axes = plot_before_after(img)
        axes[1].set_axis_off()
        axes[1].imshow(retina_mask.rpe_upper_edge(img))
        fig.suptitle(img_path.name)
        plt.tight_layout()
        file_name = srf_output_path / img_path.name
        plt.savefig(file_name, dpi=400, bbox_inches='tight')
        # plt.show()
        plt.close()
