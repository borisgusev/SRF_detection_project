import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import get_file_paths

from skimage import feature
from skimage import color
from skimage import filters
from skimage import util

from scipy.ndimage import gaussian_filter
from scipy import ndimage

import image_edit_utils as utls
import cv2


# *OPTIONAL
def apply_brightness_contrast(input_img, brightness=0, contrast=0):

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


# Read in the image

healthy, srf = get_file_paths.get_all_train_data()
srf = [srf[0]]

for img_path in srf:
    image_original = plt.imread(img_path)

    # Remove white frame around the image (interfere with edge detection, the edges are detected on the border)
    image_original = color.rgba2rgb(image_original)
    image_original = utls.rm_white_frame(image_original)

    plt.subplot(221)
    plt.title("Original")
    plt.imshow(image_original)

    # improve quality of image
    #image_enhanced = apply_brightness_contrast(image_original, brightness = 10, contrast = 15)
    image_gray = color.rgb2gray(image_original)

    gaus_par = 3
    image_gray_gaus = ndimage.gaussian_filter(image_gray, gaus_par)

    plt.subplot(222)
    plt.title("Gaussian smoothing with par =" + str(gaus_par))
    plt.imshow(image_gray_gaus)

    # Segmentation
    # cluster_labels is a matrix of image shape,
    # where each cell value corresponds to the segment number of pixel,
    # i.e. 0 - a pixel of darkest segment; (nclust-1) - a pixel of the brightest segment
    # nclust - number of clusters
    n = 7
    segmented_image, cluster_labels = utls.segmentation(image_gray_gaus,
                                                        nclust=n)

    plt.subplot(223)
    plt.title("Segmentation with nclust = " + str(n))
    plt.imshow(segmented_image)

    sigma_val = 3
    # edge detection
    edges = feature.canny(image=image_gray_gaus, sigma=sigma_val)

    # # get rif of the topmost edge
    # for j in range(edges.shape[1]):
    #     for i in range(edges.shape[0]):
    #         if (edges[i,j]==True):
    #             edges[i,j]=False
    #             break

    plt.subplot(224)
    plt.title("Edge detection with sigma = " + str(sigma_val))
    plt.imshow(edges, cmap='gray')

    plt.show()

    segmented_image_with_edges = np.copy(segmented_image)

    y_edges = np.where(edges)[0]
    x_edges = np.where(edges)[1]

    for i in range(len(y_edges)):
        if (y_edges[i] != 0 or x_edges[i] != 0
                or y_edges[i] != segmented_image_with_edges.shape[0]
                or x_edges[i] != segmented_image_with_edges.shape[1]):
            segmented_image_with_edges[y_edges[i], x_edges[i]] = (
                1, 0, 0)  # possibly will help......

    # Plot Segmented image with detected edges
    plt.subplot(121)
    plt.title("Original")
    plt.imshow(image_original)

    plt.subplot(122)
    plt.title("Segmented image with edges")
    plt.imshow(segmented_image_with_edges)

    plt.show()
