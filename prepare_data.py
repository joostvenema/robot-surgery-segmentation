"""
Split train images in train/valid/test (60/20/20)
"""
from pathlib import Path

from tqdm import tqdm
from random import shuffle
from shutil import copyfile
import cv2
import numpy as np


data_path = Path('data')

train_path = data_path / 'train'

binary_factor = 255

if __name__ == '__main__':

    (train_path / 'train').mkdir(exist_ok=True, parents=True)
    (train_path / 'valid').mkdir(exist_ok=True, parents=True)
    (train_path / 'test').mkdir(exist_ok=True, parents=True)

    file_names = list((train_path).glob('*_sat.jpg'))
    shuffle(file_names)
    train_files = file_names[0:int(len(file_names)/10*6)]
    valid_files = file_names[int(len(file_names)/10*6):int(len(file_names)/10*8)]
    test_files = file_names[int(len(file_names)/10*8):]
    
    print('separate training data')
    for file_name in tqdm(train_files):
        mask_file =  (file_name.stem).replace('_sat', '_mask') + '.png'
        dsm_file = (file_name.stem).replace('_sat', '_dsm') + '.jpg'
        copyfile(str(file_name),  str((train_path / 'train' / (file_name.stem + '.jpg'))))
        copyfile(str(train_path / mask_file), str(train_path / 'train' / mask_file))
        copyfile(str(train_path / dsm_file), str(train_path / 'train' / dsm_file))

    print('separate validation data')
    for file_name in tqdm(valid_files):
        mask_file =  (file_name.stem).replace('_sat', '_mask') + '.png'
        dsm_file = (file_name.stem).replace('_sat', '_dsm') + '.jpg'
        copyfile(str(file_name),  str((train_path / 'valid' / (file_name.stem + '.jpg'))))
        copyfile(str(train_path / dsm_file), str(train_path / 'valid' / dsm_file))
        copyfile(str(train_path / mask_file), str(train_path / 'valid' / mask_file))

    print('separate test data')
    for file_name in tqdm(test_files):
        mask_file =  (file_name.stem).replace('_sat', '_mask') + '.png'
        dsm_file = (file_name.stem).replace('_sat', '_dsm') + '.jpg'
        copyfile(str(file_name),  str((train_path / 'test' / (file_name.stem + '.jpg'))))
        copyfile(str(train_path / dsm_file), str(train_path / 'test' / dsm_file))
        copyfile(str(train_path / mask_file), str(train_path / 'test' / mask_file))
