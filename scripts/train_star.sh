#!/usr/bin/env bash
# DeepFashion
python train.py --dataroot ./fashion_data/ --name fashion_PInet_cycle --model PInet --dataset_mode keypoint --n_layers 3 --norm instance --batchSize 6 --pool_size 0 --resize_or_crop no --gpu_ids 0 --BP_input_nc 18 --no_flip --which_model_netG PInet --sepiter 50 --niter 40 --niter_decay 20 --cycle_gan --checkpoints_dir ./checkpoints --pairLst ./fashion_data/fasion-resize-pairs-train.csv --L1_type l1_plus_perL1 --n_layers_D 3 --with_D_PP 1 --with_D_PB 1  --display_id 1
