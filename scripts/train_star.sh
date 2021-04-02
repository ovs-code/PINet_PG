#!/usr/bin/env bash
# DeepFashion
lr=0.0002
name='fashion_PInet_exp'

python train.py --dataroot ./fashion_data/ --name $name --model PInet --dataset_mode keypoint --n_layers 3 --norm instance --batchSize 6 --pool_size 0 --resize_or_crop no --gpu_ids 0 --BP_input_nc 17 --which_model_netG PInet --sepiter 0 --niter 1 --niter_decay 0 --checkpoints_dir ./checkpoints --L1_type l1_plus_perL1 --n_layers_D 3 --with_D_PP 1 --with_D_PB 1  --display_id 1 --max_dataset_size 100 --lr $lr
