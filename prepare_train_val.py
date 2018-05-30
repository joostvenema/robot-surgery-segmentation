from dataset import data_path

def get_filelists(mode='train'):

    if mode == 'train':
        train_path = data_path / 'train' / 'train'
        train_valid_path = data_path / 'train' / 'valid'
        return list((train_path.glob('*.jpg'))), list((train_valid_path.glob('*.jpg')))

    elif mode == 'test':
        test_path = data_path / 'train' / 'test'
        return [], list((test_path.glob('*.jpg')))

    else:
        valid_path = data_path / 'valid'
        return [], list((valid_path.glob('*.jpg')))
