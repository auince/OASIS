from os import PathLike
from pathlib import Path

import numpy as np
from PIL import Image


ISAID_MODEL_IDS = (
    'deeplabv3plus_r50_d8',
    'pspnet_r50_d8',
    'hrnet_w48_fcn',
    'deeplabv3plus_r18_d8',
    'pspnet_r18_d8',
)
SPACENET_MODEL_IDS = (
    'sam_road',
    'pfr_dpn92_model01',
    'rngdet',
    'rngdet_plus_plus',
    'sat2graph',
)

_ISAID_PALETTE = (
    (0, 0, 0), (0, 0, 63), (0, 63, 63), (0, 63, 0),
    (0, 63, 127), (0, 63, 191), (0, 63, 255), (0, 127, 63),
    (0, 127, 127), (0, 0, 127), (0, 0, 191), (0, 0, 255),
    (0, 191, 127), (0, 127, 191), (0, 127, 255), (0, 100, 155),
)


def _read_map(value, *, isaid, ground_truth):
    if not isinstance(value, (str, PathLike)):
        return np.asarray(value)
    path = Path(value)
    suffix = path.suffix.lower()
    if suffix == '.npy':
        return np.load(path, allow_pickle=False)
    if suffix == '.npz' and isaid:
        with np.load(path, allow_pickle=False) as archive:
            return archive['segmentation']
    if suffix != '.png':
        raise ValueError(f'Unsupported mask format: {path}')
    with Image.open(path) as image:
        if isaid and ground_truth and image.mode == 'RGB':
            colors = np.asarray(image)
            labels = np.full(colors.shape[:2], 255, dtype=np.uint8)
            for label, color in enumerate(_ISAID_PALETTE):
                labels[np.all(colors == color, axis=-1)] = label
            if (labels == 255).any():
                raise ValueError(f'Unknown iSAID GT palette color: {path}')
            return labels
        if image.mode not in ('1', 'L', 'I', 'I;16', 'I;16L', 'I;16B') and not (isaid and image.mode == 'P'):
            raise ValueError(f'Mask PNG must contain a single channel: {path}')
        return np.asarray(image)


def _pair(prediction, ground_truth, *, isaid):
    pred = _read_map(prediction, isaid=isaid, ground_truth=False)
    gt = _read_map(ground_truth, isaid=isaid, ground_truth=True)
    if pred.ndim != 2 or gt.ndim != 2 or pred.shape != gt.shape or not gt.size:
        raise ValueError('Prediction and ground truth must share a nonempty 2-D grid')
    return pred, gt


def isaid_score(prediction, ground_truth):
    pred, gt = _pair(prediction, ground_truth, isaid=True)
    for grid in (pred, gt):
        if grid.dtype.kind not in 'iu' or np.any((grid < 0) | (grid > 15)):
            raise ValueError('iSAID class maps must contain integer IDs from 0 through 15')
    foreground = gt > 0
    total = int(np.count_nonzero(foreground))
    if not total:
        raise ValueError('iSAID foreground accuracy is undefined without GT foreground')
    return int(np.count_nonzero((pred == gt) & foreground)) / total


def spacenet_score(prediction, ground_truth):
    pred, gt = _pair(prediction, ground_truth, isaid=False)
    for grid in (pred, gt):
        if grid.dtype.kind not in 'biuf' or not np.isin(grid, (0, 1, 255)).all():
            raise ValueError('SpaceNet3 masks must be binary (0/1 or 0/255)')
    roads = gt != 0
    if not roads.any():
        raise ValueError('SpaceNet3 GT must contain road pixels')
    predicted_roads = pred != 0
    tp = int(np.count_nonzero(predicted_roads & roads))
    union = int(np.count_nonzero(predicted_roads | roads))
    return tp / union
