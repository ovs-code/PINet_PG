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

INPUT_SIZE = (384, 288)
TRESHOLD = 0.05
SCALE_FACTOR = 4


def pad_image(image: Image, target_size: Tuple[int, int], color='white') -> Image:
    "Pad the input image to the left and bottom"
    width, height = image.size
    if width > target_size[0] or height > target_size[1]:
        raise ValueError('Image too large in at least one dimension.')

    padded = Image.new('RGB', target_size, color)
    padded.paste(image, (0, 0))
    return padded


class PoseEstimator:

    def __init__(self, cfg, use_cuda):
        self.model = models.pose_hrnet.get_pose_net(cfg, False)
        if use_cuda:
            self.model = self.model.cuda()
        self.model.load_state_dict(torch.load(
            cfg.TEST.MODEL_FILE), strict=False)
        self.model.eval()
        self.use_cuda = use_cuda
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

    def infer(self, image: Image):
        """Takes an input RGB image <= INPUT_SIZE and returns an (17 x 2)-array"""
        width, height = image.size
        if image.size != INPUT_SIZE:
            image = pad_image(image, INPUT_SIZE)
        batch = self.transform(image).unsqueeze(0)
        if self.use_cuda:
            batch = batch.cuda()
        with torch.no_grad():
            output = self.model(batch).detach().cpu().numpy()
        preds, maxvals = get_final_preds(output)
        points = (preds[0] * SCALE_FACTOR).astype(int)[..., ::-1]
        maxvals = maxvals[0]
        invalid = (points[..., 1] >= width) | (
            points[..., 0] >= height) | (maxvals[..., 0] < TRESHOLD)
        points[invalid] = -1
        return points

    def infer_batch(self, images):
        width, height = images[0].size
        if images[0].size != INPUT_SIZE:
            images = [pad_image(image, INPUT_SIZE) for image in images]
        batch = torch.stack([self.transform(image) for image in images])
        if self.use_cuda:
            batch = batch.cuda()
        with torch.no_grad():
            output = self.model(batch).detach().cpu().numpy()
        preds, maxvals = get_final_preds(output)
        points = (preds * SCALE_FACTOR).astype(int)[..., ::-1]
        invalid = (points[..., 1] >= width) | (
            points[..., 0] >= height) | (maxvals[..., 0] < TRESHOLD)
        points[invalid] = -1
        return points


if __name__ == '__main__':
    # most important parameters
    input_folder = './fashion_data/train/'
    output_path = './fashion_data/fasion-resize-annotation-train.csv'
    pose_estimator = 'assets/pretrains/pose_hrnet_w48_384x288.pth'

    batch_size = 128

    args = Namespace(
        cfg=pkg_resources.resource_filename(
            'hrnet_pose', 'yaml/coco/hrnet/w48_384x288_adam_lr1e-3.yaml'),
        dataDir='.',
        logDir='.',
        modelDir='.',
        opts=['TEST.MODEL_FILE', pose_estimator],
        prevModelDir='.'
    )
    update_config(cfg, args)

    model = PoseEstimator(cfg, use_cuda=True)

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
