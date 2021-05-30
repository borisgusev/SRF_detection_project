from skimage import color, feature, filters, segmentation, morphology, measure
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from sklearn import cluster
from tqdm import tqdm
from pathlib import Path

import image_preprocessing
import get_file_paths
import find_blobs
import image_edit_utils as utls
import retina_mask

def segmentation(gray_image, nclust=3): 

    image = color.gray2rgb(gray_image)

    x, y, z = image.shape
    image_2d = image.reshape(x*y, z)
    image_2d.shape

    kmeans_cluster = cluster.KMeans(n_clusters=nclust)
    kmeans_cluster.fit(image_2d)
    cluster_centers = kmeans_cluster.cluster_centers_
    cluster_labels = kmeans_cluster.labels_
    image_segmented = cluster_centers[cluster_labels].reshape(x, y, z)

    cluster_labels_matrix = cluster_labels.reshape(x, y)

    return (image_segmented, cluster_labels_matrix)

def sort_labels(segmented_image, lables):
    label_brightness = []
    for lable in range(np.max(lables)+1):
        label_brightness.append(np.max(segmented_image[lables == lable]))
    sorted_labels = np.argsort(label_brightness)
    return sorted_labels

if __name__ == '__main__':
    healthy, srf = get_file_paths.get_all_train_data()

    img = plt.imread(srf[0])
    img = image_preprocessing.preprocess(img)
    # blur = filters.gaussian(img, sigma=15)
    # edges = feature.canny(img, sigma = 5)

    mask = retina_mask.retinal_mask(img)

    seg_img, cluster_labels = segmentation(img, nclust = 3)
    # modified = seg_img
    modified = cluster_labels != 0
    # modified = ndimage.binary_fill_holes(modified)



    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    # plt.subplot(1, 3, 2)
    # plt.imshow(mask, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(modified, cmap='gray')
    plt.show()

    # fig, axes = find_blobs.plot_before_after(img)
    # plt.tight_layout()
    # plt.show()

    # fix, ax = filters.try_all_threshold(modified)
    # plt.show()
    output_path = Path('segmentation_output')
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
        # mask = retina_mask.retinal_mask(img)

        blurred = filters.gaussian(img, sigma = 1)
        seg_img, cluster_labels = segmentation(blurred, nclust = 6)
        sorted_labels = sort_labels(seg_img, cluster_labels)
        # modified = cluster_labels == sorted_labels[1]
        modified = seg_img
        # modified = ndimage.binary_fill_holes(modified)



        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        # plt.subplot(1, 3, 2)
        # plt.imshow(mask, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(modified, cmap='gray')
        plt.tight_layout()
        file_name = healthy_output_path / img_path.name
        plt.savefig(file_name, dpi=400, bbox_inches='tight')
        plt.close()

    srf_output_path = output_path / 'SRF'
    srf_output_path.mkdir(exist_ok=True)
    for img_path in tqdm(srf):
        img = plt.imread(img_path)
        img = image_preprocessing.preprocess(img)
        # mask = retina_mask.retinal_mask(img)

        blurred = filters.gaussian(img, sigma = 1)
        seg_img, cluster_labels = segmentation(blurred, nclust = 6)
        sorted_labels = sort_labels(seg_img, cluster_labels)
        # modified = cluster_labels == sorted_labels[1]
        modified = seg_img
        # modified = ndimage.binary_fill_holes(modified)



        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        # plt.subplot(1, 3, 2)
        # plt.imshow(mask, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(modified, cmap='gray')
        plt.tight_layout()
        file_name = srf_output_path / img_path.name
        plt.savefig(file_name, dpi=400, bbox_inches='tight')
        plt.close()