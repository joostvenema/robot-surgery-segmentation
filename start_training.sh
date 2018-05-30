#!/bin/bash
python3 train.py --device-ids 0,1 --batch-size 6 --workers 4 --lr 0.0001 --n-epochs 20 --jaccard-weight 1 --model UNet16
python3 train.py --device-ids 0,1 --batch-size 6 --workers 4 --lr 0.00001 --n-epochs 40 --jaccard-weight 1 --model UNet16
python3 train.py --device-ids 0,1 --batch-size 6 --workers 4 --lr 0.000001 --n-epochs 60 --jaccard-weight 1 --model UNet16
