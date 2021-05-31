from skimage import color 
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster
from tqdm import tqdm
from pathlib import Path

import image_preprocessing
import get_img_paths

def segmentation(gray_image, nclust=3):
    """Applies K-Means segmentation to gray image, returning segmented image and labels matrix

    The matrix contains labels in range(nclust) in the order of brightness

    Args:
        gray_image (Image Array): grayscale image
        nclust (int, optional): number of means to be used in K-means. Defaults to 3.

    Returns:
        (Img, Label_Array): Tuple of segmented image and label array
    """
    image = color.gray2rgb(gray_image)

    x, y, z = image.shape
    image_2d = image.reshape(x*y, z)
    image_2d.shape

    kmeans_cluster = cluster.KMeans(n_clusters=nclust)
    kmeans_cluster.fit(image_2d)
    cluster_centers = kmeans_cluster.cluster_centers_
    cluster_labels = kmeans_cluster.labels_
    segmented_img = cluster_centers[cluster_labels].reshape(x, y, z)
    cluster_labels_matrix = cluster_labels.reshape(x, y)

    # relabel the clusters such that 0 is darkest and nclust-1 is brightest
    sorted_labels = _sort_labels(segmented_img, cluster_labels_matrix)
    new_labels_matrix = np.zeros_like(cluster_labels_matrix)
    for i in range(nclust):
        new_labels_matrix[cluster_labels_matrix == sorted_labels[i]] = i
    cluster_labels_matrix = new_labels_matrix
    return segmented_img, cluster_labels_matrix

def _sort_labels(segmented_image, lables):
    """Returns a list of labels in the order of brightness

    Args:
        segmented_image (Image Array): Segmented Image
        lables (Array): Array of labels for each pixel

    Returns:
        [int]: List of labels in the order of brightness
    """
    label_brightness = []
    for lable in range(np.max(lables)+1):
        label_brightness.append(np.max(segmented_image[lables == lable]))
    sorted_labels = np.argsort(label_brightness)
    return sorted_labels

if __name__ == '__main__':
    # Script to run segmentation on all training images
    output_path = Path('segmentation_output')
    output_path.mkdir(exist_ok=True)
    healthy, srf = get_img_paths.train_data()

    nclust = 6

    healthy_output_path = output_path / 'NoSRF'
    healthy_output_path.mkdir(exist_ok=True)
    for img_path in tqdm(healthy):
        img = plt.imread(img_path)
        img = image_preprocessing.preprocess(img)
        seg_img, cluster_labels = segmentation(img, nclust = nclust)
        # Uncomment one below if want segmented image or particular segment
        # modified = seg_img
        modified = cluster_labels == 1

        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(modified, cmap='gray')
        plt.suptitle(img_path.name)
        plt.tight_layout()
        file_name = healthy_output_path / img_path.name
        plt.savefig(file_name, dpi=400, bbox_inches='tight')
        plt.close()

    srf_output_path = output_path / 'SRF'
    srf_output_path.mkdir(exist_ok=True)
    for img_path in tqdm(srf):
        img = plt.imread(img_path)
        img = image_preprocessing.preprocess(img)
        seg_img, cluster_labels = segmentation(img, nclust = nclust)
        # Uncomment one below if want segmented image or particular segment
        # modified = seg_img
        modified = cluster_labels == 1

        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(modified, cmap='gray')
        plt.suptitle(img_path.name)
        plt.tight_layout()
        file_name = healthy_output_path / img_path.name
        plt.savefig(file_name, dpi=400, bbox_inches='tight')
        plt.close()