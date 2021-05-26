from skimage import color, feature, filters, segmentation, morphology, measure
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

import image_preprocessing
import get_file_paths
import find_blobs
import image_edit_utils as utls
import retina_mask

healthy, srf = get_file_paths.get_all_train_data()

img = plt.imread(srf[0])
img = image_preprocessing.preprocess(img)
# blur = filters.gaussian(img, sigma = 5)
# edges = feature.canny(img, sigma = 5)


mask = retina_mask.retinal_mask(img)

# seg_img, cluster_labels = utls.segmentation(img)
seg_labels = segmentation.slic(color.gray2rgb(img), mask = mask, enforce_connectivity=True,
                               n_segments=50,
                               compactness=20,
                               start_label=1)
modified = color.label2rgb(seg_labels, img)


plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.subplot(1,3,2)
plt.imshow(mask, cmap='gray')
plt.subplot(1, 3, 3)
plt.imshow(modified, cmap='gray')
plt.show()

# fig, axes = find_blobs.plot_before_after(img)
# plt.tight_layout()
# plt.show()

# fix, ax = filters.try_all_threshold(modified)
# plt.show()