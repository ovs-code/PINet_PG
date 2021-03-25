# Adapted from tengteng95/Pose-Transfer

import numpy as np
import pandas as pd
import json
import os

MISSING_VALUE = -1


def load_pose_cords_from_strings(y_str, x_str):
    y_cords = json.loads(y_str)
    x_cords = json.loads(x_str)
    return np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)


def cords_to_map(cords, img_size, sigma=6):
    """Convert a list of keypoint locations to 17 different heatmaps for each keypoint."""
    result = np.zeros(img_size + cords.shape[0:1], dtype='uint8')
    for i, point in enumerate(cords):
        if point[0] == MISSING_VALUE or point[1] == MISSING_VALUE:
            continue
        xx, yy = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
        result[..., i] = np.exp(-((yy - point[0]) ** 2 + (xx - point[1]) ** 2) / (2 * sigma ** 2))
        # result[..., i] = np.where(((yy - point[0]) ** 2 + (xx - point[1]) ** 2) < (sigma ** 2), 1, 0)
    return result


def compute_pose(image_dir, annotations_file, savePath, sigma=6):
    annotations_file = pd.read_csv(annotations_file, sep=':')
    annotations_file = annotations_file.set_index('name')
    image_size = (256, 176)
    cnt = len(annotations_file)
    for i in range(cnt):
        print('processing %d / %d ...' % (i, cnt))
        row = annotations_file.iloc[i]
        name = row.name
        print(savePath, name)
        file_name = os.path.join(savePath, name + '.npy')
        kp_array = load_pose_cords_from_strings(
            row.keypoints_y, row.keypoints_x)
        pose = cords_to_map(kp_array, image_size, sigma)
        np.save(file_name, pose)
        # input()


if __name__ == '__main__':
    # fix PATH
    img_dir = 'test_data'  # raw image path
    annotations_file = 'test_data/test-annotation-test.csv'  # pose annotation path
    save_path = 'test_data/testK'  # path to store pose maps
    compute_pose(img_dir, annotations_file, save_path)
