import json
from pathlib import Path

import numpy as np
from PIL import Image


MODELS = (
    'yolop_official_bdd100k', 'yolopv2_official_bdd100k',
    'hybridnets_official_bdd100k', 'ayolom_s_official_bdd100k',
    'ayolom_n_official_bdd100k',
)
VEHICLE_CLASSES = frozenset(('car', 'bus', 'truck', 'train'))


def _mask(path):
    with Image.open(path) as image:
        values = np.asarray(image)
    if values.ndim == 3 and values.shape[-1] in (3, 4):
        rgb = values[..., :3]
        if not np.all(rgb == rgb[..., :1]):
            raise ValueError(f'Expected a grayscale BDD100K mask: {path}')
        values = rgb[..., 0]
    if values.ndim != 2:
        raise ValueError(f'Invalid BDD100K mask shape: {path}')
    return values > 0


def ground_truth_loader(detections, masks):
    with Path(detections).open(encoding='utf-8') as stream:
        rows = json.load(stream)
    by_name = {row['name']: row['labels'] for row in rows}
    if len(by_name) != len(rows):
        raise ValueError('Duplicate BDD100K annotation names')

    def load(source):
        if not isinstance(source, str) or not source or Path(source).name != source:
            raise ValueError('Expected the original BDD100K filename')
        boxes = {}
        for label in by_name[source]:
            if 'box2d' in label and label['category'] in VEHICLE_CLASSES:
                boxes.setdefault(label['category'], []).append(
                    [label['box2d'][key] for key in ('x1', 'y1', 'x2', 'y2')])
        boxes = {category: np.asarray(values, dtype=np.float64).reshape(-1, 4)
                 for category, values in boxes.items()}
        mask_name = Path(source).with_suffix('.png').name
        return (boxes,
                _mask(Path(masks) / 'drivable' / mask_name),
                _mask(Path(masks) / 'lane' / mask_name))

    return load


def _prediction(path, model):
    if model not in MODELS:
        raise ValueError(f'Unknown BDD100K model: {model}')
    with np.load(path, allow_pickle=False) as data:
        if not {'boxes_xyxy', 'confidences', 'classes'}.issubset(data.files):
            raise ValueError(f'Class-resolved detections are required for per-class IoU; vehicle-only caches cannot be used: {path}')
        boxes, confidence, classes = data['boxes_xyxy'], data['confidences'], data['classes']
        road, lane = data['drivable_mask'], data['lane_mask']
        class_names = data['class_names'] if 'class_names' in data.files else None
    if boxes.shape == confidence.shape == classes.shape == (0,):
        boxes = boxes.reshape(0, 4)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or confidence.shape != (len(boxes),) or classes.shape != (len(boxes),):
        raise ValueError(f'Unaligned BDD100K detection arrays: {path}')
    if not all(np.isfinite(array).all() for array in (boxes, confidence)):
        raise ValueError(f'Nonfinite BDD100K detection: {path}')
    if np.any((confidence < 0) | (confidence > 1)):
        raise ValueError(f'Invalid BDD100K confidence: {path}')
    keep = confidence >= 0.25
    if classes.dtype.kind in 'US':
        names = classes.astype('U')
    elif classes.dtype.kind in 'iuf':
        if not np.isfinite(classes).all() or np.any(classes < 0) or np.any(classes != np.floor(classes)):
            raise ValueError(f'Invalid BDD100K class ID: {path}')
        if class_names is None:
            raise ValueError(f'Explicit semantic class_names are required for numeric IDs; vehicle-only caches cannot be used: {path}')
        if class_names.ndim != 1 or class_names.dtype.kind not in 'US':
            raise ValueError(f'Expected a one-dimensional semantic class_names vocabulary: {path}')
        if np.any(classes >= len(class_names)):
            raise ValueError(f'BDD100K class ID exceeds class_names vocabulary: {path}')
        names = class_names.astype('U')[classes.astype(np.int64)]
    else:
        raise ValueError(f'Expected semantic class names or numeric class IDs: {path}')
    if any(name not in VEHICLE_CLASSES for name in names[keep]):
        raise ValueError(f'Expected explicit car, bus, truck or train labels for retained detections: {path}')
    for mask in (road, lane):
        if mask.ndim != 2 or (mask.dtype != np.bool_ and not np.isin(mask, (0, 1)).all()):
            raise ValueError(f'Expected a two-dimensional binary prediction mask: {path}')
    return boxes[keep], names[keep], road.astype(bool, copy=False), lane.astype(bool, copy=False)


def _union(boxes, shape):
    mask = np.zeros(shape, dtype=bool)
    height, width = shape
    for x1, y1, x2, y2 in boxes:
        if not np.isfinite((x1, y1, x2, y2)).all() or x2 < x1 or y2 < y1:
            raise ValueError('Invalid xyxy box')
        if x2 == x1 or y2 == y1:
            continue
        left = int(np.clip(np.floor(x1), 0, width))
        right = int(np.clip(np.ceil(x2), 0, width))
        top = int(np.clip(np.floor(y1), 0, height))
        bottom = int(np.clip(np.ceil(y2), 0, height))
        mask[top:bottom, left:right] = True
    return mask


def _iou(prediction, ground_truth):
    if prediction.shape != ground_truth.shape:
        raise ValueError('BDD100K prediction and GT grids differ')
    union = np.count_nonzero(prediction | ground_truth)
    return np.count_nonzero(prediction & ground_truth) / union if union else 0.0


def driving_score(prediction, ground_truth, model):
    boxes, classes, road, lane = _prediction(prediction, model)
    gt_boxes, gt_road, gt_lane = ground_truth
    if gt_road.shape != gt_lane.shape:
        raise ValueError('BDD100K GT grids differ')
    if not isinstance(gt_boxes, dict) or set(gt_boxes) - VEHICLE_CLASSES:
        raise ValueError('Class-resolved car, bus, truck and train GT boxes are required')
    categories = sorted(set(classes) | {category for category, values in gt_boxes.items() if len(values)})
    by_class = {
        category: _iou(_union(boxes[classes == category], gt_road.shape),
                       _union(gt_boxes.get(category, ()), gt_road.shape))
        for category in categories
    }
    detection = sum(by_class.values()) / len(by_class) if by_class else 0.0
    drivable = _iou(road, gt_road)
    lane = _iou(lane, gt_lane)
    return {'task_score': (detection + drivable + lane) / 3,
            'detection_iou': detection, 'detection_iou_by_class': by_class,
            'drivable_iou': drivable, 'lane_iou': lane}
