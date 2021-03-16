from options.infer_options import InferOptions
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from keras.models import load_model

from models.PINet20 import TransferModel, create_model
from tool import cords_to_map, get_coords
from util import util

IMAGE_SIZE = (256, 176)


class InferencePipeline:
    def __init__(self, pose_estimator, pinet: TransferModel, segmentator):
        self.pose_estimator = pose_estimator
        self.pinet = pinet
        self.segmentator = segmentator
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __call__(self, image: Image, target_pose_map: np.array) -> Image:
        # get pose
        imgBGR = np.array(image)[:, :, ::-1]
        pose = get_coords(imgBGR, self.pose_estimator)

        # convert to pose map
        pose_map = cords_to_map(pose, IMAGE_SIZE)

        # get segmentation map ...
        spl_onehot = self.segmentator.get_segmap(image)

        # run PINet
        image_norm = self.transform(image)
        output_image, output_segmentation = self.pinet.infer(
            image_norm, pose_map, target_pose_map, spl_onehot)
        return output_image, output_segmentation


class DummySegmentationModel:
    def __init__(self, path):
        self.path = path

    def get_segmap(self, *args):
        num_class = 12
        SPL_path = self.path
        SPL_img = Image.open(SPL_path)
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


def load_pose_from_file(path):
    pose_img = np.load(path)
    pose = torch.from_numpy(pose_img).float()  # h, w, c
    pose = pose.transpose(2, 0)  # c,w,h
    pose = pose.transpose(2, 1)  # c,h,w
    return pose


if __name__ == '__main__':
    SOURCE_IMAGE_PATH = 'test_data/test/randomphoto_small.jpg'
    TEST_SEG_PATH = 'test_data/testSPL2/randomphoto_small.png'
    TARGET_POSE_PATH = 'test_data/test/Oskar_pad_small.jpg.npy'
    OUPUT_PATH = 'test_data/out.jpg'

    opt = InferOptions().parse()
    pinet = create_model(opt)
    pose_estimator = load_model()
    segmentator = DummySegmentationModel(TEST_SEG_PATH)
    pipeline = InferencePipeline(pose_estimator, pinet, segmentator)

    source_image = Image.open(SOURCE_IMAGE_PATH)
    target_pose = load_pose_from_file(TARGET_POSE_PATH)

    output_image, output_segmentation = pipeline(source_image, target_pose)

    Image.fromarray(util.tensor2im(output_image)).save(OUPUT_PATH)
