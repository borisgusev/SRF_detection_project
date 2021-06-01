Signal and Image Processing Project to detect Sub-Retinal Fluid (SRF)


The general method:
    1. Preprocess image (color space conversion, denoising)
    2. Apply Laplacian of Gaussians to find round dark regions
    3. Use K-means segmentation to define layer of interest (RPE)
    4. Filter dark regions, keeping only those that are directly above the RPE
    5. If any detected dark regions remain, classify image as having SRF


Script Descriptions:
    main.py                     Main script to generate csv file of image names and corresponding labels for test-data set

    find_blobs.py               Functions related to finding dark areas and filtering them

    get_img_paths.py            Script to fetch test and train data sets

    image_preprocessing.py      image preprocessing functions

    retina_mask.py              functions to create binary masks

    segmentation.py             functions related to K-means segmentation

    train_data_analysis.py      quick script to test detection on train-data set