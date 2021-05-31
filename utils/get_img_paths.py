from pathlib import Path


def train_data(path_str = 'Train-Data'):
    path = Path(path_str)
    return _get_healthy_data(path), _get_srf_data(path)

def _get_healthy_data(path):
    healthy_path = path / 'NoSRF'
    healthy_images = list(healthy_path.glob('*.png'))
    return healthy_images


def _get_srf_data(path):
    srf_path = path / 'SRF'
    srf_images = list(srf_path.glob('*.png'))
    return srf_images

def test_data(path_str = 'Test-Data'):
    path = Path(path_str)
    img_paths = list(path.glob('*png'))
    return img_paths