from typing import List
import numpy as np
import os
from PIL import Image
import numba


def load_backgrounds(path: str) -> List[np.ndarray]:
    return [
        np.array(Image.open(os.path.join(path, file)))
        for file in os.listdir(path)
        if file.endswith('.jpg')
    ]

@numba.jit
def background_swap(image, segmentation, background):
    new = np.copy(image)
    for col in range(segmentation.shape[0]):
        for row in range(segmentation.shape[1]):
            if segmentation[col,row] == 0:
                new[col,row,0] = background[col,row,0]
                new[col,row,1] = background[col,row,1]
                new[col,row,2] = background[col,row,2]
    return new
