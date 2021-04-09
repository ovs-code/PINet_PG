import asyncio
import time
import torch

from mlq.queue import MLQ

# imports for the keypoint model
from hrnet_pose.config import cfg, update_config
from hrnet_pose import models
import pkg_resources
from argparse import Namespace

from pipeline import InferencePipeline
from options.infer_options import InferOptions

from PIL import Image
from io import BytesIO
import base64

mlq = MLQ('pose_transfer', 'localhost', 6379, 0)

# Create the Inference Pipeline instance
pip_opts = InferOptions().parse(['--name', 'fashion_PInet_cycle'])
INFERENCE_PIPELINE = InferencePipeline.from_opts(pip_opts)

def inference(input_dict, *args):
    """
    Function for the actual inference
    """
    # unpack the input
    source_image = input_dict['source_image']
    target_pose = input_dict['target_pose']

    # TODO: bring the target pose list into the 
    # right input format for the torch model
    # transform the target pose into tensor
    target_pose = torch.Tensor(target_pose)
    
    # get the source pose
    # source_pose = KEYPOINT_MODEL(source_image)

    # get the source segmentation
    # source_segmentation = SEGMENTATION_MODEL(source_image)

    # do the final transfer
    source_image = Image.open(BytesIO(base64.b64decode(source_image)))
    target_image = INFERENCE_PIPELINE(source_image, target_pose)
    target_image_file = BytesIO()
    target_image.save(target_image_file, format="PNG")
    target_image = base64.b64encode(target_image_file.getvalue()).decode()
    return {'target_image': target_image}

def main():
    print("Running, waiting for messages.")
    async def doit():
        mlq.create_listener(inference)
    asyncio.get_event_loop().run_until_complete(doit())

if __name__ == '__main__':
    main()
