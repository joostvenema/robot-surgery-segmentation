"""
Script generates predictions, splitting original images into tiles, and assembling prediction back together
"""
import argparse
from prepare_train_val import get_filelists
from dataset import DeepglobeDataset
import cv2
from models import UNet16, LinkNet34, UNet11, UNet
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
import utils
import prepare_data
from torch.utils.data import DataLoader
from torch.nn import functional as F

from transforms import (ImageOnly,
                        Normalize,
                        DualCompose)

img_transform = DualCompose([
    ImageOnly(Normalize())
])


def get_model(model_path, model_type='UNet16', problem_type='binary'):
    """

    :param model_path:
    :param model_type: 'UNet', 'UNet16', 'UNet11', 'LinkNet34'
    :return:
    """

    num_classes = 1

    if model_type == 'UNet16':
        model = UNet16(num_classes=num_classes)
    elif model_type == 'UNet11':
        model = UNet11(num_classes=num_classes)
    elif model_type == 'LinkNet34':
        model = LinkNet34(num_classes=num_classes)
    elif model_type == 'UNet':
        model = UNet(num_classes=num_classes)

    state = torch.load(str(model_path))
    state = {key.replace('module.', ''): value for key, value in state['model'].items()}
    model.load_state_dict(state)

    if torch.cuda.is_available():
        return model.cuda()

    model.eval()

    return model


def predict(model, from_file_names, batch_size: int, to_path, problem_type):
    loader = DataLoader(
        dataset=DeepglobeDataset(from_file_names, transform=img_transform, mode='predict', problem_type=problem_type),
        shuffle=False,
        batch_size=batch_size,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available()
    )

    for batch_num, (inputs, paths) in enumerate(tqdm(loader, desc='Predict')):
        inputs = utils.variable(inputs, volatile=True)

        outputs = model(inputs)

        for i, image_name in enumerate(paths):

            factor = prepare_data.binary_factor
            t_mask = (F.sigmoid(outputs[i, 0]).data.cpu().numpy() * factor).astype(np.uint8)

            h, w = t_mask.shape

            full_mask = t_mask

            _, full_mask = cv2.threshold(full_mask, 127, 255, cv2.THRESH_BINARY)

            full_mask = cv2.cvtColor(full_mask, cv2.COLOR_GRAY2RGB)

            mask_folder = Path(paths[i]).parent.parent.name

            (to_path / mask_folder).mkdir(exist_ok=True, parents=True)

            cv2.imwrite(str(to_path / mask_folder / (Path(paths[i]).stem + '.png')), full_mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model_path', type=str, default='data/models', help='path to model folder')
    arg('--model_file', type=str, default='model_dg_3.pt', help='filename of trained model')
    arg('--model_type', type=str, default='UNet11', help='network architecture',
        choices=['UNet', 'UNet11', 'UNet16', 'LinkNet34'])
    arg('--output_path', type=str, help='path to save images', default='.')
    arg('--batch-size', type=int, default=4)
    arg('--workers', type=int, default=8)
    arg('--mode', type=str, default='valid', choices=['valid', 'test'], help='Which dataset to predict')

    args = parser.parse_args()

    _, file_names = get_filelists(args.mode)
    model = get_model(str(Path(args.model_path).joinpath(args.model_file)),
                      model_type=args.model_type, problem_type='binary')

    print('num file_names = {}'.format(len(file_names)))

    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    predict(model, file_names, args.batch_size, output_path, problem_type='binary')
