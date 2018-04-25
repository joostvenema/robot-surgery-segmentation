from dataset import data_path


def get_split(fold):
    folds = {0: [1, 3],
             1: [2, 5],
             2: [4, 8],
             3: [6, 7]}

    train_path = data_path / 'train'

    train_file_names = []
    val_file_names = []

    for image_id in range(1, 9):
        if image_id in folds[fold]:
            val_file_names += list((train_path / ('instrument_dataset_' + str(instrument_id)) / 'images').glob('*'))
        else:
            train_file_names += list((train_path / ('instrument_dataset_' + str(instrument_id)) / 'images').glob('*'))

    return train_file_names, val_file_names

def get_filelists(mode='train'):
    
    file_names = []

    if mode == 'train':
        train_path = data_path / 'train'
    elif mode == 'test':
        train_path = data_path / 'test'
    else:
        train_path = data_path / 'valid'
    
    file_names += list((train_path.glob('*.jpg')))

    if mode == 'train':
        # Get 60% for test, 20% for validation
        return file_names[0:int(len(file_names)/10*6)], file_names[int(len(file_names)/10*6):int(len(file_names)/10*8)]
    else:
        return [], file_names
