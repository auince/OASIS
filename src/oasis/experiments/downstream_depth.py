from pathlib import Path

import numpy as np


EIGEN_CROP = (slice(45, 471), slice(41, 601))
NYU_SHAPE = (480, 640)
DEPTH_MODEL_IDS = (
    "depth_anything_v2_metric_indoor_small",
    "zoedepth_n",
    "metric3d_v2_vit_small",
    "unidepth_v2_vitl14",
    "depth_pro",
)


def load_depth_map(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if "depth" not in data.files:
                raise ValueError("Depth NPZ must contain a depth array")
            return data["depth"]
    raise ValueError("Expected metric depth .npy or .npz with key depth")


def depth_score(prediction, ground_truth):
    pred = np.asarray(prediction)
    truth = np.asarray(ground_truth)
    if pred.shape != NYU_SHAPE or truth.shape != NYU_SHAPE:
        raise ValueError("Expected aligned, uncropped NYU depth maps of shape (480, 640)")
    if pred.dtype.kind not in "iuf" or truth.dtype.kind not in "iuf":
        raise ValueError("Expected real numeric depth maps in meters")
    pred = pred[EIGEN_CROP].astype(np.float64, copy=False)
    truth = truth[EIGEN_CROP]
    valid = np.isfinite(truth) & (truth > 1e-3) & (truth < 10.0)
    count = np.count_nonzero(valid)
    if not count:
        raise ValueError("No valid ground-truth pixels within the NYU Eigen crop")
    pred = pred[valid]
    truth = truth[valid].astype(np.float64, copy=False)
    positive = np.isfinite(pred) & (pred > 0.0)
    pred = pred[positive]
    truth = truth[positive]
    with np.errstate(over="ignore"):
        ratio = np.maximum(pred / truth, truth / pred)
    return float(np.count_nonzero(ratio < 1.25) / count)
