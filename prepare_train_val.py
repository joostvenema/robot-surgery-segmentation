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

def get_filelists():

    train_path = data_path / 'train'

    train_file_names = []
    val_file_names = []
    train_file_names += list((train_path.glob('[0-7]*.jpg')))

    val_file_names += list((train_path.glob('[8-9]*.jpg')))


    return train_file_names, val_file_names
