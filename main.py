import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import get_img_paths
import find_blobs
import image_preprocessing

if __name__ == '__main__':
    output_filename = 'gusev_solis.csv'
    col_names = ['filename', 'labels']
    output = pd.DataFrame(columns = col_names)
    test_imgs_paths = get_img_paths.test_data()

    img_names = [path.name for path in test_imgs_paths]

    labels = []
    for img_path in tqdm(test_imgs_paths):
        img = plt.imread(img_path)
        img = image_preprocessing.preprocess(img)

        blobs = find_blobs.find_dark_blobs(img)
        blobs = find_blobs.filter_blobs(img, blobs)

        # if any blobs present, write 1, else 0
        if len(blobs) != 0:
            labels.append(1)
        else:
            labels.append(0)
    
    # add the data to respective columns in DataFrame
    output.filename = img_names
    output.labels = labels

    output.to_csv(output_filename, index = False)
