#!/usr/bin/env bash
# DeepFashion
lr=0.0002
name='fashion_PInet_exp'

python train.py --dataroot ./fashion_data/ --name $name --batchSize 6 --gpu_ids 0 --sepiter 1 --niter 1 --niter_decay 0 --display_id 1 --lr $lr --max_dataset_size 50
