from pathlib import Path


def train_data(path_str='Train-Data'):
    """Returns a lists of healthy and srf image paths from training data folder

    Args:
        path_str (str, optional): path to folder containing SRF and NoSRF folders with images. Defaults to 'Train-Data'.

    Returns:
        ([Path], [Path]): tuple of two lists; each list containing paths for images
    """
    path = Path(path_str)
    return _get_healthy_data(path), _get_srf_data(path)


def test_data(path_str='Test-Data'):
    """Returns a list of test data images

    Args:
        path_str (str, optional): path towards test data folder. Defaults to 'Test-Data'.

    Returns:
        [Path]: list of image paths
    """
    path = Path(path_str)
    img_paths = list(path.glob('*png'))
    return img_paths


def _get_healthy_data(path):
    healthy_path = path / 'NoSRF'
    healthy_images = list(healthy_path.glob('*.png'))
    return healthy_images


def _get_srf_data(path):
    srf_path = path / 'SRF'
    srf_images = list(srf_path.glob('*.png'))
    return srf_images
