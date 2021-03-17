from __future__ import annotations

import numpy as np
import torch
from keras.models import load_model
from PIL import Image
from torchvision import transforms

from models.PINet20 import TransferModel, create_model
from options.infer_options import InferOptions
from tool import cords_to_map, get_coords, reorder_pose, load_pose_from_file
from util import util

IMAGE_SIZE = (256, 176)


class InferencePipeline:
    def __init__(self, pose_estimator, pinet: TransferModel, segmentator, opt):
        """Initialize the pipeline with already loaded models."""
        self.pose_estimator = pose_estimator
        self.pinet = pinet
        self.segmentator = segmentator
        self.opt = opt
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    @classmethod
    def from_opts(cls, opt) -> InferencePipeline:
        """Load all trained models required from the locations indicated in opt."""
        TEST_SEG_PATH = 'test_data/testSPL2/randomphoto_small.png'

        pinet = create_model(opt).eval()
        pose_estimator = load_model(opt.pose_estimator, compile=False)
        segmentator = DummySegmentationModel(TEST_SEG_PATH)
        return cls(pose_estimator, pinet, segmentator, opt)

    def __call__(self, image: Image, target_pose_map: torch.Tensor) -> Image:
        # get pose
        imgBGR = np.array(image)[:, :, ::-1]
        pose = get_coords(imgBGR, self.pose_estimator)

        # convert to pose map
        pose_map = reorder_pose(cords_to_map(pose, IMAGE_SIZE))

        # get segmentation map ...
        spl_onehot = self.segmentator.get_segmap(image).unsqueeze(0)

        # run PINet
        image_norm = self.transform(image).unsqueeze(0)

        if self.opt.gpu_ids:
            # move data to GPU
            device = self.opt.gpu_ids[0]
            pose_map = pose_map.cuda(device)
            target_pose_map = target_pose_map.cuda(device)
            image_norm = image_norm.cuda(device)
            spl_onehot = spl_onehot.cuda(device)

        output_image, output_segmentation = self.pinet.infer(
            image_norm, pose_map, target_pose_map, spl_onehot)
        return Image.fromarray(util.tensor2im(output_image))


class DummySegmentationModel:
    def __init__(self, path):
        self.path = path

    def get_segmap(self, *args):
        num_class = 12
        SPL_path = self.path
        SPL_img = Image.open(SPL_path)
        if np.array(SPL_img).shape[1]==256:
            SPL_img = SPL_img.crop((40, 0, 216, 256))
        SPL_img = SPL_img.transpose(Image.FLIP_LEFT_RIGHT)
        SPL_img = np.expand_dims(np.array(SPL_img), 0)
        _, h, w = SPL_img.shape
        tmp = torch.from_numpy(SPL_img).view(-1).long()
        ones = torch.sparse.torch.eye(num_class)
        ones = ones.index_select(0, tmp)
        SPL_onehot = ones.view([h, w, num_class])
        SPL_onehot = SPL_onehot.permute(2, 0, 1)
        # SPL = torch.from_numpy(SPL_img).long()
        return SPL_onehot


if __name__ == '__main__':
    SOURCE_IMAGE_PATH = 'test_data/test/randomphoto_small.jpg'
    TARGET_POSE_PATH = 'test_data/testK/randomphoto_small.jpg.npy'
    OUPUT_PATH = 'test_data/out.jpg'

    opt = InferOptions().parse()
    pipeline = InferencePipeline.from_opts(opt)

    source_image = Image.open(SOURCE_IMAGE_PATH)
    target_pose = load_pose_from_file(TARGET_POSE_PATH)

    output_image = pipeline(source_image, target_pose)

    output_image.save(OUPUT_PATH)
