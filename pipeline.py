from __future__ import annotations

import math
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms, io
from torchvision.io import video

from models.PINet20 import TransferModel, create_model
from options.infer_options import InferOptions
from tool import cords_to_map, reorder_pose
from tool.compute_coordinates import DEFAULT_ARGS, PoseEstimator
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
        TEST_SEG_PATH = 'test_data/testSPL2/fashionMENDenimid0000537801_7additional.png'

        pinet = create_model(opt).eval()
        args = DEFAULT_ARGS
        args.opts = ['TEST.MODEL_FILE', opt.pose_estimator]
        pose_estimator = PoseEstimator(args, opt.gpu_ids != [])
        segmentator = DummySegmentationModel(TEST_SEG_PATH)
        return cls(pose_estimator, pinet, segmentator, opt)

    def __call__(self, image: Image, target_pose_map: torch.Tensor) -> Image:
        # get pose
        pose = self.pose_estimator.infer(image)

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
        with torch.no_grad():
            output_image, _ = self.pinet.infer(
                image_norm,
                pose_map,
                target_pose_map,
                spl_onehot
            )
        return Image.fromarray(util.tensor2im(output_image))

    def render_video(self, image: Image, target_poses: str, batch_size=16):
        # get pose estimation
        pose = self.pose_estimator.infer(image)
        pose_map = reorder_pose(cords_to_map(pose, IMAGE_SIZE))
        # get semantic segmentation
        spl_onehot = self.segmentator.get_segmap(image).unsqueeze(0)
        # transform image
        image_norm = self.transform(image).unsqueeze(0)
        # read target poses
        pose_files = [os.path.join(target_poses, fname) for fname in sorted(os.listdir(target_poses))]
        target_poses = torch.cat([reorder_pose(np.load(file)) for file in pose_files])
        # create batch(es) of the same image, source pose, segmentation and different target poses
        if self.opt.gpu_ids:
            device = self.opt.gpu_ids[0]
            pose_map = pose_map.cuda(device)
            target_poses = target_poses.cuda(device)
            image_norm = image_norm.cuda(device)
            spl_onehot = spl_onehot.cuda(device)

        pose_map = pose_map.expand(batch_size, -1, -1, -1)
        spl_onehot = spl_onehot.expand(batch_size, -1, -1, -1)
        image_norm = image_norm.expand(batch_size, -1, -1, -1)
        with torch.no_grad():
            nframes = target_poses.size(0)
            nbatches = math.ceil(nframes / batch_size)
            for i in range(nbatches):
                nel = min(batch_size, nframes-i*batch_size)
                output_images, output_segmentations = self.pinet.infer(
                    image_norm[:nel],
                    pose_map[:nel],
                    target_poses[i*batch_size : (i+1)*batch_size],
                    spl_onehot[:nel]
                )
                yield (output_images.detach().cpu(), output_segmentations.detach().cpu())


class DummySegmentationModel:
    def __init__(self, path):
        self.path = path

    def get_segmap(self, *args):
        num_class = 12
        SPL_path = self.path
        SPL_img = Image.open(SPL_path)
        if np.array(SPL_img).shape[1] == 256:
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

    SOURCE_IMAGE_PATH = 'test_data/test/fashionMENDenimid0000537801_7additional.jpg'
    TARGET_POSE_PATH = 'test_data/testK/fashionMENDenimid0000537801_7additional.jpg.npy'
    OUPUT_PATH = 'test_data/out.jpg'

    opt = InferOptions().parse()
    print(opt)
    exit()
    pipeline = InferencePipeline.from_opts(opt)


    # target_pose = load_pose_from_file(TARGET_POSE_PATH)
    # output_image = pipeline(source_image, target_pose)
    import time
    st = time.time()
    videos = [io.read_video('test_data/seq.mp4', pts_unit='sec')[0]]
    segs = [torch.zeros_like(videos[0], dtype=torch.uint8)]
    images = [torch.zeros_like(videos[0], dtype=torch.uint8)]
    for person in ['fashionMENDenimid0000537801_7additional']:
        source_image = Image.open(f'test_data/test/{person}.jpg')
        pipeline.segmentator.path = f'test_data/testSPL2/{person}.png'
        frames, segmentations = zip(*pipeline.render_video(source_image, 'test_data/seq/'))
        frames = torch.cat(frames)
        frames = frames.float()
        frames = torch.movedim(frames, 1, 3)
        frames = (frames + 1) / 2.0 * 255.0
        videos.append(frames.byte())
        segmentations = torch.cat(segmentations)
        segmentations = torch.stack([torch.from_numpy(util.tensor2im(torch.argmax(sf, axis=0, keepdim=True).data, True)) for sf in segmentations])
        segs.append(segmentations.byte())
        source_image_tensor = torch.from_numpy(np.array(source_image)).unsqueeze(0).expand(frames.size())
        images.append(source_image_tensor)
        print(time.time() - st)

    comp_video = torch.cat([torch.cat(part, dim=2) for part in (images, segs, videos)], dim=1)
    io.write_video('test_data/out.mp4', comp_video, fps=30)
    print(time.time() - st)

    # output_image.save(OUPUT_PATH)
