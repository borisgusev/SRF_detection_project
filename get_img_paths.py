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



def get_healthy_data():
    # train_data_path = Path('Train-Data')
    train_data_path = Path('Train-Data')
    healthy_path = train_data_path / 'NoSRF'
    # healthy_images = [file for file in healthy_path.iterdir()]
    healthy_images = list(healthy_path.glob('*.png'))
    return healthy_images


def get_srf_data():
    # train_data_path = Path('Train-Data')
    train_data_path = Path('Train-Data')
    srf_path = train_data_path / 'SRF'
    # srf_images = [file for file in srf_path.iterdir()]
    srf_images = list(srf_path.glob('*.png'))
    return srf_images


# only training data
def train_data():
    return get_healthy_data(), get_srf_data()
