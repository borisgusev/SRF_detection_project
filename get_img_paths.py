from pathlib import Path


# def get_test_data (path_str='Test-Data'):
#     """Returns a lists of test image paths

#     Args:
#         path_str (str, optional): path to folder containing SRF and NoSRF folders with images. Defaults to 'Train-Data'.

#     Returns:
#         ([Path], [Path]): tuple of two lists; each list containing paths for images
#     """

def get_test_data(path_str='Test-Data'):
    """Returns a list of test data images

    Args:
        path_str (str, optional): path towards test data folder. Defaults to 'Test-Data'.

    Returns:
        [Path]: list of image paths
    """
    path = Path(path_str)
    img_paths = list(path.glob('*png'))
    return img_paths