from pathlib import Path



def get_healthy_data():
    train_data_path = Path('Train-Data')
    healthy_path = train_data_path / 'NoSRF'
    # healthy_images = [file for file in healthy_path.iterdir()]
    healthy_images = list(healthy_path.glob('*.png'))
    return healthy_images


def get_srf_data():
    train_data_path = Path('Train-Data')
    srf_path = train_data_path / 'SRF'
    # srf_images = [file for file in srf_path.iterdir()]
    srf_images = list(srf_path.glob('*.png'))
    return srf_images


def get_all_train_data():
    return get_healthy_data(), get_srf_data()


if __name__ == '__main__':
    # quick check to see if the file paths and the number of each image is correct
    healthy, srf = get_all_train_data()
    print(len(healthy))
    print(healthy[0])
    print(len(srf))
    print(srf[0])