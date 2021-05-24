import get_file_paths
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, color, feature

healthy, srf = get_file_paths.get_all_train_data()

n = 9
img = plt.imread(srf[n])
img = color.rgba2rgb(img)
modified = color.rgb2gray(img)
modified = filters.sato(
    modified,
    sigmas=range(1, 10, 1),
    black_ridges=False,
)
# modified = feature.canny(modified)

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original')

plt.subplot(1, 2, 2)
plt.imshow(modified, cmap='gray')
plt.title('Modified')

plt.show()
