#!/usr/bin/env bash
# DeepFashion

# number of samples to evaluate
how_many=4
# set to 0 for GPU testing
gpu_id=0

python test.py --dataroot ./fashion_data/ --name fashion_PInet --model PInet --phase test --dataset_mode keypoint --norm instance --batchSize 1 --resize_or_crop no --gpu_ids $gpu_id --BP_input_nc 17 --no_flip --which_model_netG PInet --checkpoints_dir ./checkpoints --which_epoch test --results_dir ./results --how_many $how_many
