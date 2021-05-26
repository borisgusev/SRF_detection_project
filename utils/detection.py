import matplotlib.pyplot as plt
import numpy as np

import get_file_paths, image_preprocessing, find_blobs

if __name__ == '__main__':
    healthy, srf = get_file_paths.get_all_train_data()



    no_srf_detected = 0
    for img_path in healthy:
        img = plt.imread(img_path)
        img = image_preprocessing.preprocess(img)
        blobs = find_blobs.find_candidate_srf_blobs(img)
        blobs = find_blobs.filter_blob_candidates(img, blobs)
        
        print(len(blobs))
        if len(blobs) == 0:
            no_srf_detected += 1
        
    print(f'In {no_srf_detected} healthy images no SRF was detected.')