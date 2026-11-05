from numbers import Integral

import numpy as np


MEDICAL_MODEL_IDS = (
    'microsoft-biomedclip',
    'medclip-vit',
    'pmc-clip',
    'unimed-clip-vit-b16',
    'pubmed-clip-vit-base-patch32',
)

MEDICAL_SUBSET_CLASS_COUNTS = {
    'chestmnist_224': 14,
    'dermamnist_224': 7,
    'octmnist_224': 4,
    'pathmnist_224': 9,
    'pneumoniamnist_224': 2,
}


def _vector(value, name):
    array = np.asarray(value)
    if array.ndim != 1 or array.size == 0 or array.dtype.kind not in 'iuf':
        raise ValueError(f'{name} must be a nonempty numeric vector')
    if not np.isfinite(array).all():
        raise ValueError(f'{name} must contain finite values')
    return array


def medical_score(prediction, ground_truth):
    probabilities = _vector(prediction, 'Prediction').astype(np.float64)
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError('Probabilities must be in [0, 1]')
    if isinstance(ground_truth, Integral) and not isinstance(ground_truth, (bool, np.bool_)):
        if len(probabilities) not in (2, 4, 7, 9):
            raise ValueError('Unexpected single-label probability count')
        if not 0 <= ground_truth < len(probabilities):
            raise ValueError('Ground-truth class index is out of range')
        if not np.isclose(probabilities.sum(), 1.0, rtol=0.0, atol=1e-5):
            raise ValueError('Single-label probabilities must sum to one')
        return float(probabilities[ground_truth])
    labels = _vector(ground_truth, 'Ground truth')
    if labels.size != 14 or probabilities.size != 14 or not np.isin(labels, (0, 1)).all():
        raise ValueError('ChestMNIST requires 14 binary labels and 14 positive-state probabilities')
    return float(np.mean(np.where(labels == 1, probabilities, 1.0 - probabilities)))
