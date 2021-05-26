import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import util
from skimage import feature
from skimage import color
from skimage import filters

from scipy import ndimage
from sklearn import cluster

# def edge_detection (gray_scaled_image, edges, threshold=0.1):
#     segmented_image = segmentation(gray_scaled_image, nclust=7)
#     segmented_image = color.rgb2gray(segmented_image)

#     h = gray_scaled_image.shape[0]
#     hpx = int(h*threshold)
#     print (hpx)
#     error = 0.001
#     target_segment = np.max(segmented_image) # target layer
#     print (target_segment)
#     for i in range (edges.shape[0]):
#         for j in range (edges.shape[1]):
#             if (edges[i,j]==True):
#                 status = False
#                 for k in range (i-hpx,i):
#                    if (np.abs(segmented_image[k,j]-target_segment)<error):
#                        status = True
#                        break
#                 for k in range (i,i+hpx):
#                    if (np.abs(segmented_image[k,j]-target_segment)<error):
#                        status = True
#                        break
#                 edges[i,j]=status

#     return (edges)


# Remove white frame
def rm_white_frame(im):

    # create a negative (to invert the white frame into black, to fill with 0)
    im = 1 - im

    h, w, d = im.shape
    #left limit
    for i in range(w):
        if np.sum(im[:, i, :]) > 0:
            break
    #right limit
    for j in range(w - 1, 0, -1):
        if np.sum(im[:, j, :]) > 0:
            break

    #top limit
    for k in range(h):
        if np.sum(im[k, :, :]) > 0:
            break
    #bottom limit
    for l in range(h - 1, 0, -1):
        if np.sum(im[l, :, :]) > 0:
            break

    cropped = im[k:l + 1, i:j + 1, :].copy()
    #back to normal
    cropped = 1 - cropped

    return (cropped)


def segmentation(gray_image, nclust=7):

    image = color.gray2rgb(gray_image)

    x, y, z = image.shape
    image_2d = image.reshape(x * y, z)
    image_2d.shape

    kmeans_cluster = cluster.KMeans(n_clusters=nclust)
    kmeans_cluster.fit(image_2d)
    cluster_centers = kmeans_cluster.cluster_centers_
    cluster_labels = kmeans_cluster.labels_
    image_segmented = cluster_centers[cluster_labels].reshape(x, y, z)

    cluster_labels_matrix = cluster_labels.reshape(x, y)

    return (image_segmented, cluster_labels_matrix)


# def filter_edges(edges, gray_image):
#     print (np.max(gray_image))
#     for i in range(edges.shape[0]):
#         for j in range(edges.shape[1]):
#             if (i-1>0 and i<edges.shape[0]-1):
#                 if (gray_image[i-1,j]<130 and gray_image[i+1,j]<130):
#                     edges[i,j]=False

#     return (edges)
