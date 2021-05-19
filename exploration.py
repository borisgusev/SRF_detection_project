import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import feature
from skimage import color
from skimage import filters

train_data = Path('Train-Data')
healthy_images = [img_path for img_path in (train_data / 'NoSRF').iterdir()]
srf_images = [img_path for img_path in (train_data / 'SRF').iterdir()]

img = plt.imread(healthy_images[0])
img = plt.imread(srf_images[9])
img = color.rgba2rgb(img)
img = color.rgb2gray(img)
edge = filters.sobel(filters.gaussian(img, sigma=5))

# img = color.rgb2gray(img)
# edge = filters.sobel_v(img)

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(edge, cmap='gray')
plt.show()