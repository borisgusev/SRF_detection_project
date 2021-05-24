import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import matplotlib
from matplotlib import pyplot as plt
from pathlib import Path

from skimage import feature
from skimage import color
from skimage import filters
from skimage import util

import os
from scipy.ndimage import gaussian_filter
import image_edit_utils as utls
from scipy import ndimage


# *OPTIONAL
def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

# Read in the image

# assumes that the unzipped training folder is included in the directory
train_data = Path('Train-Data')
# healthy_images = [img_path for img_path in ('NoSRF').iterdir()]
srf_images = [img_path for img_path in (train_data / 'SRF').iterdir()]

train_data = Path('Train-Data')

# placeholder to view individual images
# image_original = plt.imread(healthy_images[0])
image_original = plt.imread(srf_images[9])


#image_original = plt.imread("SRF/input_1647_1.png")
# Remove white frame around
image_original = utls.rm_white_frame(image_original)

plt.subplot(221)
plt.title("Original")
plt.imshow(image_original)


# improve quality of image
#image_enhanced = apply_brightness_contrast(image_original, brightness = 10, contrast = 15)
image_gray = color.rgba2rgb(image_original)
image_gray = color.rgb2gray(image_gray)

gaus_par=3
image_gray_gaus = ndimage.gaussian_filter(image_gray, gaus_par)

utls.segmentation(image_gray_gaus)

plt.subplot(222)
plt.title("Gaussian smoothing with par ="+str(gaus_par))
plt.imshow(image_gray_gaus)



# # Sobel filter

#image_gray_gaus = image_gray_gaus.astype('int32')

# sx = ndimage.sobel(image_gray_gaus, axis=0, mode="mirror")
# sy = ndimage.sobel(image_gray_gaus, axis=1, mode="mirror")
# image_gray_gaus_sobel = np.hypot(sx, sy)
# image_gray_gaus_sobel *= 255.0 / np.max(image_gray_gaus_sobel) # normalization 0..255
# image_gray_gaus_sobel = np.float32(image_gray_gaus_sobel)

segmented_image = utls.segmentation(image_gray_gaus)

plt.subplot(223)
plt.title("Segmentation")
plt.imshow(segmented_image)

# image_gray_gaus = np.float32(image_gray_gaus)

sigma_val=3
edges = feature.canny(image=image_gray_gaus, sigma = sigma_val)  

plt.subplot(224)
plt.title("Edge detection with sigma = "+str(sigma_val))
plt.imshow(edges,cmap='gray')

plt.show()


y_edges=np.where(edges)[0]
x_edges=np.where(edges)[1]


for i in range(len(y_edges)):
    if (y_edges[i]!=0 or x_edges[i]!=0 or y_edges[i]!=segmented_image.shape[0] or x_edges[i]!=segmented_image.shape[1]):
        segmented_image[y_edges[i],x_edges[i]]=(1,0,0)

plt.subplot(121)
plt.title("Original")
plt.imshow(image_original)

plt.subplot(122)
plt.title("Segmented image with edges")
plt.imshow(segmented_image)

plt.show()




