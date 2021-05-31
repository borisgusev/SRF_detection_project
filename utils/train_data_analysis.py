import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import get_img_paths
import image_preprocessing
import find_blobs

if __name__ == '__main__':
    '''Small script to count detected srf in each training data set
    '''
    healthy, srf = get_img_paths.train_data()

    no_srf_detected = 0
    for img_path in tqdm(healthy):
        img = plt.imread(img_path)
        img = image_preprocessing.preprocess(img)
        blobs = find_blobs.find_dark_blobs(img)
        blobs = find_blobs.filter_blobs(img, blobs)

        if len(blobs) == 0:
            no_srf_detected += 1

    print(f'Negative result for SRF was correctly returned for {no_srf_detected} out of {len(healthy)} healthy images')


    srf_detected = 0
    for img_path in tqdm(srf):
        img = plt.imread(img_path)
        img = image_preprocessing.preprocess(img)
        blobs = find_blobs.find_dark_blobs(img)
        blobs = find_blobs.filter_blobs(img, blobs)

        if len(blobs) != 0:
            srf_detected += 1

    
    print(f'Positive result for SRF was correctly returned for {srf_detected} out of {len(srf)} srf images')

        
