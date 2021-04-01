#!/usr/bin/env bash
# DeepFashion

python train.py --dataroot ./fashion_data/ --name fashion_PInet_cycle --model PInet --dataset_mode keypoint --n_layers 3 --norm instance --batchSize 6 --pool_size 0 --resize_or_crop no --gpu_ids 0 --BP_input_nc 17 --no_flip --which_model_netG PInet --sepiter 1 --niter 1 --niter_decay 1 --checkpoints_dir ./checkpoints --pairLst ./fashion_data/fasion-resize-pairs-train.csv --L1_type l1_plus_perL1 --n_layers_D 3 --with_D_PP 1 --with_D_PB 1  --display_id 1