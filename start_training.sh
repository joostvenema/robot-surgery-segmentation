#!/bin/bash
python3 train.py --device-ids 0,1 --batch-size 6 --workers 4 --lr 0.0001 --n-epochs 15 --type binary --jaccard-weight 1 --model UNet16
python3 train.py --device-ids 0,1 --batch-size 6 --workers 4 --lr 0.00001 --n-epochs 15 --type binary --jaccard-weight 1 --model UNet16
python3 train.py --device-ids 0,1 --batch-size 6 --workers 4 --lr 0.000001 --n-epochs 30 --type binary --jaccard-weight 1 --model UNet16
