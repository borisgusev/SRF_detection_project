from skimage import color, feature, filters, segmentation, morphology, measure
from skimage import io, exposure
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

import image_preprocessing
import get_file_paths
import find_blobs
import image_edit_utils as utls
import retina_mask

from scipy.ndimage import label
from scipy.ndimage import generate_binary_structure

from sklearn import cluster
from bilateral import filter_bilateral


def segmentation(image, nclust=3): 

    x, y, z = image.shape
    image_2d = image.reshape(x*y, z)
    image_2d.shape

    kmeans_cluster = cluster.KMeans(n_clusters=nclust,  n_init=10)
    kmeans_cluster.fit(image_2d)
    cluster_centers = kmeans_cluster.cluster_centers_
    cluster_labels = kmeans_cluster.labels_
    image_segmented = cluster_centers[cluster_labels].reshape(x, y, z)

    label_colors = cluster_centers[:,0]
    cluster_labels_matrix = cluster_labels.reshape(x, y)

    return (image_segmented, cluster_labels_matrix, label_colors)



healthy, srf = get_file_paths.get_all_train_data()

#img = plt.imread(srf[2])
for img in srf:
    img = plt.imread(img)
    img = image_preprocessing.preprocess(img)
    mask = retina_mask.retinal_mask(img)
    # blur = filters.gaussian(img, sigma=3) # Gaussian (edges are not well visible)
    blur = filter_bilateral( img, 5.0, 0.2) # Bilateral filter works MUCH better, but slow :'(
    blur[np.where(mask==False)]=0


    img_clustered, seg_labels, label_colors = segmentation(color.gray2rgb(blur), nclust=3)
    img_clustered = color.rgb2gray(img_clustered)
    img_clustered[np.where(mask==False)]=0 # get rid of the background

    # edges = feature.canny(blur, sigma=3)
    # edges[np.where(mask==False)]=False


    plt.subplot(1, 2, 1)
    plt.imshow(blur, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(img_clustered, cmap='gray')


    plt.show()
