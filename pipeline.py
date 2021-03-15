from PIL import Image
import numpy as np

from tool import get_coords, cords_to_map

IMAGE_SIZE = (256, 176)

class InferencePipeline:
    def __init__(self, pose_estimator, pinet):
        self.pose_estimator = pose_estimator
        self.pinet = pinet

    def infer(self, image: Image, pose: np.array) -> Image:
        imgBGR = np.array(image)[:, :, ::-1]
        # get pose,
        pose = get_coords(imgBGR, self.pose_estimator)
        # convert to pose map,
        # TODO: check if the format/type of pose matches the input of cords_to_map
        cords_to_map(pose, IMAGE_SIZE)

        # get segmentation map ...

        # run PINet
        # TODO: create batch from single sample
        # self.pinet.set_input(data)
        # self.pinet.test()
        # TODO: self.pinet.get_output()

        raise NotImplementedError
