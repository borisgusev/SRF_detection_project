from skimage import color, feature
import image_preprocessing
import get_file_paths
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

healthy, srf = get_file_paths.get_all_train_data()

img = plt.imread(srf[0])
img = image_preprocessing.preprocess(img)

edge = feature.canny(img, sigma = 5)

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(edge, cmap='gray')
plt.show()