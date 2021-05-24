import get_file_paths
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import color, util, exposure, feature


def find_candidate_srf_blobs(img):
    img = color.rgba2rgb(img)
    img = color.rgb2gray(img)
    img = util.invert(img)
    img = exposure.adjust_gamma(img, gamma=2.5)
    blobs = feature.blob_log(img,
                             min_sigma=3,
                             max_sigma=15,
                             num_sigma=20,
                             threshold=0.18,
                             exclude_border=(65))
    blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
    return blobs


def filter_blob_candidates():
    pass


def plot_blobs(axes, blobs):
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
        axes.add_patch(c)
    # return axes


def plot_before_after(img):
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(img)
    ax[0].set_title('Original')
    ax[1].imshow(img)
    ax[1].set_title('Candidate Blobs')
    blobs = find_candidate_srf_blobs(img)
    plot_blobs(ax[1], blobs)
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    return fig, axes

if __name__ == '__main__':
    output_path = Path('blob_output')
    output_path.mkdir(exist_ok=True)
    healthy, srf = get_file_paths.get_all_train_data()

    healthy_output_path = output_path / 'NoSRF'
    healthy_output_path.mkdir(exist_ok=True)
    for img_path in healthy:
        img = plt.imread(img_path)
        fig, axes = plot_before_after(img)
        fig.suptitle(img_path.name)
        plt.tight_layout()
        file_name = healthy_output_path / img_path.name
        plt.savefig(file_name, dpi = 400, bbox_inches = 'tight')
        # plt.show()


    srf_output_path = output_path / 'SRF'
    srf_output_path.mkdir(exist_ok=True)
    for img_path in srf:
        img = plt.imread(img_path)
        fig, axes = plot_before_after(img)
        fig.suptitle(img_path.name)
        plt.tight_layout()
        file_name = srf_output_path / img_path.name
        plt.savefig(file_name, dpi = 400, bbox_inches = 'tight')
        # plt.show()
