# Adapted from https://github.com/tengteng95/Pose-Transfer/blob/master/tool/calPCKH_fashion.py

import argparse
import ast
from typing import List, Tuple

import pandas as pd

MISSING_VALUE = -1
PARTS_SEL = [0, 1, 14, 15, 16, 17]
ALPHA = 0.5

parser = argparse.ArgumentParser(prog=__file__)
parser.add_argument('target_annotation')
parser.add_argument('pred_annotation')
args = parser.parse_args()

target_annotation = args.target_annotation
pred_annotation = args.pred_annotation


def is_right(px: int, py: int, tx: int, ty: int, hz: int, alpha) -> bool:
    """
        hz: head size
        alpha: norm factor
        px, py: predict coords
        tx, ty: target coords
    """
    if -1 in (px, py, tx, ty):
        return 0

    return abs(px - tx) < hz[0] * alpha and abs(py - ty) < hz[1] * alpha


def how_many_right_seq(px: List[int], py: List[int], tx: List[int], ty: List[int], *args) -> int:
    return sum(is_right(*points, *args) for points in zip(px, py, tx, ty))


def valid_points(tx: List) -> int:
    """Number of values different from MISSING_VALUE in tx."""
    return sum(v != MISSING_VALUE for v in tx)


def get_head_wh(x_coords: List[int], y_coords: List[int]) -> Tuple[int, int]:
    """Compute the size of the head, this is used for the tolerance of right predictions."""
    save_components = []
    for component in PARTS_SEL:
        if x_coords[component] == MISSING_VALUE or y_coords[component] == MISSING_VALUE:
            continue
        else:
            save_components.append([x_coords[component], y_coords[component]])

    if len(save_components) >= 2:
        x_cords, y_cords = zip(*save_components)
        xmin = min(x_cords)
        xmax = max(x_cords)
        ymin = min(y_cords)
        ymax = max(y_cords)
        final_w = xmax - xmin
        final_h = ymax - ymin
        return final_w, final_h
    else:
        return -1, -1


tAnno = pd.read_csv(target_annotation, sep=':')
pAnno = pd.read_csv(pred_annotation, sep=':')

nAll = 0
nCorrect = 0
for row in pAnno.iloc:
    pValues = row.values
    pname = pValues[0]
    pycords = ast.literal_eval(pValues[1])
    pxcords = ast.literal_eval(pValues[2])

    tValues = tAnno.query('name == %r' % pname).values[0]
    tycords = ast.literal_eval(tValues[1])
    txcords = ast.literal_eval(tValues[2])

    head_size = get_head_wh(txcords, tycords)
    if -1 in head_size:
        continue

    nAll = nAll + valid_points(tycords)
    nCorrect = nCorrect + \
        how_many_right_seq(pxcords, pycords, txcords,
                           tycords, head_size, ALPHA)

print('%d/%d %f' % (nCorrect, nAll, nCorrect / nAll))
