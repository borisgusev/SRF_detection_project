Signal and Image Processing Project to detect Sub-Retinal Fluid (SRF)

The general method:
    1. Preprocess image (color space conversion, denoising)
    2. Apply Laplacian of Gaussians to find round dark regions
    3. Use segmentation to define layer of interest (RPE)
    4. Filter dark regions, keeping only those that are directly above the RPE
    5. If any detected dark regions remain, classify image as having SRF