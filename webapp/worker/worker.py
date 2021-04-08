import asyncio
import time
import torch

from mlq.queue import MLQ

# imports for the keypoint model
from hrnet_pose.config import cfg, update_config
from hrnet_pose import models
import pkg_resources
from argparse import Namespace

mlq = MLQ('pose_transfer', 'localhost', 6379, 0)

# Create the DL models
# Keypoint model
args = Namespace(
    cfg=pkg_resources.resource_filename(
        'hrnet_pose', 'yaml/coco/hrnet/w48_256x192_adam_lr1e-3.yaml'
    ),
    dataDir='../..',
    logDir='../..',
    modelDir='../..',
    opts=['TEST.MODEL_FILE', 'pose_hrnet_w48_256x192.pth'],
    prevModelDir='../..'
)
update_config(cfg, args)

KEYPOINT_MODEL = models.pose_hrnet.get_pose_net(cfg, False)
KEYPOINT_MODEL.load_state_dict(torch.load(cfg.TEST.MODEL_FILE, map_location=torch.device('cpu')), strict=False)
KEYPOINT_MODEL.eval()

# Segmentation model
SEGMENTATION_MODEL = None

# Transfer model
TRANSFER_MODEL = None

def inference(input_dict, *args):
    """
    Function for the actual inference
    """
    print('Inside inference (at the worker)')
    # unpack the input
    source_image = input_dict['source_image']
    target_pose = input_dict['target_pose']
    
    # get the source pose
    # source_pose = KEYPOINT_MODEL(source_image)

    # get the source segmentation
    # source_segmentation = SEGMENTATION_MODEL(source_image)

    # do the final transfer
    # target_image = TRANSFER_MODEL.infer(source_image, source_pose, target_pose, source_segmentation)
    target_image = source_image
    return {'target_image': target_image}

def main():
    print("Running, waiting for messages.")
    async def doit():
        mlq.create_listener(inference)
    asyncio.get_event_loop().run_until_complete(doit())

if __name__ == '__main__':
    main()