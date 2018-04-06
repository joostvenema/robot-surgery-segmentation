"""
[1] Merge masks with different instruments into one binary mask
[2] Crop black borders from images and masks
"""
from pathlib import Path

from tqdm import tqdm
import cv2
import numpy as np

data_path = Path('data')

train_path = data_path / 'train'

binary_factor = 255

# if __name__ == '__main__':
#     for instrument_index in range(1, 9):
#         #instrument_folder = 'instrument_dataset_' + str(instrument_index)
#
#         (data_path / 'images').mkdir(exist_ok=True, parents=True)
#
#         #binary_mask_folder = (train_path / instrument_folder / 'binary_masks')
#         #binary_mask_folder.mkdir(exist_ok=True, parents=True)
#
#
#         #mask_folders = list((train_path).glob('*.png'))
#         # mask_folders = [x for x in mask_folders if 'Other' not in str(mask_folders)]
#
#         for file_name in tqdm(list((train_path).glob('*.jpg'))):
#             img = cv2.imread(str(file_name))
#
#             cv2.imwrite(str(cropped_train_path / instrument_folder / 'images' / (file_name.stem + '.jpg')), img,
#                         [cv2.IMWRITE_JPEG_QUALITY, 100])
#
#             mask_binary = np.zeros((old_h, old_w))
#
#             for mask_folder in mask_folders:
#                 mask = cv2.imread(str(mask_folder / file_name.name), 0)
#
#                 mask_binary += mask
#
#             mask_binary = (mask_binary[h_start: h_start + height, w_start: w_start + width] > 0).astype(
#                 np.uint8) * binary_factor
#
#             cv2.imwrite(str(binary_mask_folder / file_name.name), mask_binary)
