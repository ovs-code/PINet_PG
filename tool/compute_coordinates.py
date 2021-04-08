from collections import OrderedDict
import os
from argparse import Namespace
from typing import List, Tuple

import pkg_resources
import torch
from tqdm import tqdm
from hrnet_pose import models
from hrnet_pose.config import cfg, update_config
from hrnet_pose.core.inference import get_final_preds
from PIL import Image
from torchvision import transforms

INPUT_SIZE = 256, 192
IMAGE_SIZE = 176, 256
TRESHOLD = 0.15
SCALE_FACTOR = 4

DEFAULT_ARGS = Namespace(
    cfg=pkg_resources.resource_filename(
        'hrnet_pose', 'yaml/coco/hrnet/w48_256x192_adam_lr1e-3.yaml'),
    dataDir='.',
    logDir='.',
    modelDir='.',
    prevModelDir='.'
)

def transform_preds(p):
    # rotate -90 deg and scale to image dimensions
    p[..., 1] = (INPUT_SIZE[1] - 1 - p[..., 1]*SCALE_FACTOR) * IMAGE_SIZE[0] // INPUT_SIZE[1]
    p[..., 0] = p[..., 0] * SCALE_FACTOR * IMAGE_SIZE[1] // INPUT_SIZE[0]

class PoseEstimator:

    def __init__(self, args, use_cuda):
        update_config(cfg, args)
        self.model = models.pose_hrnet.get_pose_net(cfg, False)
        if use_cuda:
            self.model = self.model.cuda()
            self.model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
        else:
            self.model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE, map_location=torch.device('cpu')), strict=False)
        self.model.eval()
        self.use_cuda = use_cuda
        self.transform = transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Lambda(lambda t: torch.rot90(t, dims=(1, 2)))
        ])

    def infer(self, image: Image):
        """Takes an input RGB image <= INPUT_SIZE and returns an (17 x 2)-array"""
        batch = self.transform(image).unsqueeze(0)
        if self.use_cuda:
            batch = batch.cuda()
        with torch.no_grad():
            output = self.model(batch).detach().cpu().numpy()
        preds, maxvals = get_final_preds(output)
        transform_preds(preds)
        points = preds[0].astype(int)
        maxvals = maxvals[0]
        invalid = maxvals[..., 0] < TRESHOLD
        points[invalid] = -1
        return points

    def infer_batch(self, images):
        batch = torch.stack([self.transform(image) for image in images])
        if self.use_cuda:
            batch = batch.cuda()
        with torch.no_grad():
            output = self.model(batch).detach().cpu().numpy()
        preds, maxvals = get_final_preds(output)
        transform_preds(preds)
        points = preds.astype(int)
        invalid = maxvals[..., 0] < TRESHOLD
        points[invalid] = -1
        return points


if __name__ == '__main__':
    # most important parameters
    import sys
    input_folder, output_path = sys.argv[1:]
    pose_estimator = 'assets/pretrains/pose_hrnet_w48_256x192.pth'

    batch_size = 128

    args = DEFAULT_ARGS
    args.opts = ['TEST.MODEL_FILE', pose_estimator]

    model = PoseEstimator(args, use_cuda=True)

    with open(output_path, 'w') as result_file:
        processed_names = set()
        print('name:keypoints_y:keypoints_x', file=result_file)
        batch = OrderedDict()
        for image_name in tqdm(os.listdir(input_folder)):
            if '.ipynb_checkpoints' in image_name:
                continue

            image = Image.open(os.path.join(input_folder, image_name))
            batch[image_name] = image
            if len(batch) >= batch_size:
                for iname, pose_cords in zip(batch.keys(), model.infer_batch(list(batch.values()))):
                    print("%s: %s: %s" % (iname, list(pose_cords[:, 0]), list(
                        pose_cords[:, 1])), file=result_file)
                result_file.flush()
                batch = OrderedDict()
        if batch:
            for iname, pose_cords in zip(batch.keys(), model.infer_batch(list(batch.values()))):
                print("%s: %s: %s" % (iname, list(pose_cords[:, 0]), list(
                    pose_cords[:, 1])), file=result_file)
            result_file.flush()
