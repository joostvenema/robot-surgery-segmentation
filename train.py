import argparse
import json
from pathlib import Path
from validation import validation_binary, validation_multi

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.backends.cudnn

from models import UNet11, LinkNet34, UNet, UNet16
from loss import LossBinary
from dataset import DeepglobeDataset
import utils

from prepare_train_val import get_filelists

from transforms import (DualCompose,
                        TripleCompose,
                        ImageOnly,
                        DsmOnly,
                        Normalize,
                        NormalizeDsm,
                        HorizontalFlip,
                        VerticalFlip,
                        RandomRotate90)


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--jaccard-weight', default=1, type=float)
    arg('--device-ids', type=str, default='0', help='For example 0,1 to run on two GPUs')
    arg('--root', default='runs/debug', help='checkpoint root')
    arg('--batch-size', type=int, default=1)
    arg('--n-epochs', type=int, default=100)
    arg('--lr', type=float, default=0.0001)
    arg('--workers', type=int, default=8)
    arg('--model', type=str, default='UNet', choices=['UNet', 'UNet11', 'UNet16', 'LinkNet34'])

    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)

    num_classes = 1

    if args.model == 'UNet':
        model = UNet(num_classes=num_classes)
    elif args.model == 'UNet11':
        model = UNet11(num_classes=num_classes, pretrained='vgg')
    elif args.model == 'UNet16':
        model = UNet16(num_classes=num_classes, pretrained='vgg')
    elif args.model == 'LinkNet34':
        model = LinkNet34(num_classes=num_classes, pretrained=True)
    else:
        model = UNet(num_classes=num_classes, input_channels=4)

    if torch.cuda.is_available():
        if args.device_ids:
            device_ids = None # list(map(int, args.device_ids.split(',')))
        else:
            device_ids = None
        model = nn.DataParallel(model, device_ids=device_ids).cuda()


    loss = LossBinary(jaccard_weight=args.jaccard_weight)


    cudnn.benchmark = True

    def make_loader(file_names, shuffle=False, transform=None, problem_type='binary', mode='train'):
        return DataLoader(
            dataset=DeepglobeDataset(file_names, transform=transform, problem_type=problem_type, mode=mode),
            shuffle=shuffle,
            num_workers=args.workers,
            batch_size=args.batch_size,
            pin_memory=torch.cuda.is_available()
        )

    train_file_names, val_file_names = get_filelists()

    print('num train = {}, num_val = {}'.format(len(train_file_names), len(val_file_names)))

    train_transform = TripleCompose([
        HorizontalFlip(),
        VerticalFlip(),
        RandomRotate90(),
        ImageOnly(Normalize()),
        DsmOnly(NormalizeDsm())
    ])

    val_transform = DualCompose([
        ImageOnly(Normalize())
    ])

    train_loader = make_loader(train_file_names, shuffle=True, transform=train_transform, problem_type='binary', mode='train')
    valid_loader = make_loader(val_file_names, transform=val_transform, problem_type='binary', mode='valid')

    root.joinpath('params.json').write_text(
        json.dumps(vars(args), indent=True, sort_keys=True))

    valid = validation_binary


    utils.train(
        init_optimizer=lambda lr: Adam(model.parameters(), lr=lr),
        args=args,
        model=model,
        criterion=loss,
        train_loader=train_loader,
        valid_loader=valid_loader,
        validation=valid,
        num_classes=num_classes
    )


if __name__ == '__main__':
    main()
