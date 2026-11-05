import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from .common import atomic_json, correlations, file_sha256, read_jsonl, resolve_image_rows
from ..paths import resolve_path
from . import downstream_depth, downstream_driving, downstream_medical, downstream_remote


COHORTS = {
    'driving': {'bdd100k': 457},
    'depth': {'nyu_depth_v2': 126},
    'medical': {
        'chestmnist_224': 15, 'dermamnist_224': 57, 'octmnist_224': 53,
        'pathmnist_224': 75, 'pneumoniamnist_224': 30,
    },
    'remote': {'isaid': 215, 'spacenet3': 54},
}

MODELS = {
    'bdd100k': downstream_driving.MODELS,
    'nyu_depth_v2': downstream_depth.DEPTH_MODEL_IDS,
    **dict.fromkeys(COHORTS['medical'], downstream_medical.MEDICAL_MODEL_IDS),
    'isaid': downstream_remote.ISAID_MODEL_IDS,
    'spacenet3': downstream_remote.SPACENET_MODEL_IDS,
}


def _unique(rows, fields):
    result = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if any(not isinstance(value, str) or not value.strip() for value in key):
            raise ValueError(f'Invalid ID: {key}')
        if key in result:
            raise ValueError(f'Duplicate record: {key}')
        result[key] = row
    return result


def _path(value, index):
    if isinstance(value, str):
        return resolve_path(value, index)
    return value


def _manifest(rows, scenario, smoke_test):
    images = _unique(rows, ('image_id',))
    sources = {}
    targets = {}
    for (image_id,), row in images.items():
        dataset, source, level = (row[key] for key in ('dataset', 'source_id', 'level'))
        if dataset not in COHORTS[scenario] or not isinstance(source, str) or not source.strip():
            raise ValueError(f'Unexpected dataset or source for {image_id}')
        if type(level) is not int or level not in range(6):
            raise ValueError(f'Invalid degradation level for {image_id}')
        levels = sources.setdefault((dataset, source), set())
        if level in levels:
            raise ValueError(f'Duplicate source/level: {dataset}, {source}, {level}')
        levels.add(level)
        target = row['source_image_name' if dataset == 'bdd100k' else 'ground_truth']
        if targets.setdefault((dataset, source), target) != target:
            raise ValueError(f'Ground truth differs across degradation levels: {dataset}, {source}')
        digest = row['image_sha256']
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError(f'Invalid image SHA256 for {image_id}')
    counts = dict(Counter(dataset for dataset, _ in sources))
    if set(counts) != set(COHORTS[scenario]):
        raise ValueError(f'Missing datasets for {scenario}')
    if not smoke_test and counts != COHORTS[scenario]:
        raise ValueError(f'Unexpected source counts: {counts}; expected {COHORTS[scenario]}')
    if any(levels != set(range(6)) for levels in sources.values()):
        raise ValueError('Expected L0 through L5 for every source')
    return images, counts


def _composites(images, rows):
    scores = _unique(rows, ('image_id', 'method'))
    methods = sorted({method for _, method in scores})
    expected = {(image_id, method) for (image_id,) in images for method in methods}
    if set(scores) != expected:
        raise ValueError('Quality scores must cover every image for every method')
    formats = {}
    result = {}
    for (image_id, method), row in scores.items():
        if 'image_sha256' in row and row['image_sha256'] != images[image_id,]['image_sha256']:
            raise ValueError(f'Quality score input hash differs for {method}, {image_id}')
        score = row['score']
        is_composite = isinstance(score, dict)
        if formats.setdefault(method, is_composite) != is_composite:
            raise ValueError(f'Mixed scalar and dimension scores for {method}')
        if is_composite:
            applicable = images[image_id,]['dimensions']
            if not isinstance(applicable, list) or not applicable or any(
                not isinstance(dimension, str) or not dimension.strip() for dimension in applicable
            ) or len(applicable) != len(set(applicable)) or set(applicable) != set(score):
                raise ValueError(f'Applicable dimensions differ for {method}, {image_id}')
            values = [score[dimension] for dimension in applicable]
        else:
            values = [score]
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
            raise ValueError(f'Invalid quality score for {method}, {image_id}')
        result[image_id, method] = math.fsum(values) / len(values)
    return methods, result


def _correlate(scenario, methods, quality, rows):
    result = {}
    for method in methods:
        def compare(subset):
            return correlations(
                [row['task_score'] for row in subset],
                [quality[row['image_id'], method] for row in subset],
            )

        if scenario == 'remote':
            by_dataset = {
                dataset: compare([row for row in rows if row['dataset'] == dataset])
                for dataset in COHORTS['remote']
            }
            overall = {'n': len(rows)}
            for key in ('plcc', 'srcc'):
                values = [metrics[key] for metrics in by_dataset.values()]
                overall[key] = None if None in values else sum(values) / len(values)
            result[method] = {'overall': overall, 'datasets': by_dataset}
        else:
            result[method] = {'overall': compare(rows)}
    return result


def _task_score(dataset, prediction, ground_truth, model):
    if dataset == 'bdd100k':
        return downstream_driving.driving_score(prediction, ground_truth, model)
    if dataset == 'nyu_depth_v2':
        if isinstance(prediction, Path):
            prediction = downstream_depth.load_depth_map(prediction)
        if isinstance(ground_truth, Path):
            ground_truth = downstream_depth.load_depth_map(ground_truth)
        value = downstream_depth.depth_score(prediction, ground_truth)
    elif dataset in COHORTS['medical']:
        classes = downstream_medical.MEDICAL_SUBSET_CLASS_COUNTS[dataset]
        label_shape = (classes,) if dataset == 'chestmnist_224' else ()
        if np.shape(prediction) != (classes,) or np.shape(ground_truth) != label_shape:
            raise ValueError(f'Probability or label shape does not match {dataset}')
        value = downstream_medical.medical_score(prediction, ground_truth)
    elif dataset == 'isaid':
        value = downstream_remote.isaid_score(prediction, ground_truth)
    else:
        value = downstream_remote.spacenet_score(prediction, ground_truth)
    return {'task_score': float(value)}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', choices=tuple(COHORTS), required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--image-root', type=Path)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--quality-scores', type=Path, required=True)
    parser.add_argument('--expected-manifest-sha256')
    parser.add_argument('--bdd-detection', type=Path)
    parser.add_argument('--bdd-mask-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest_sha256 = file_sha256(args.manifest)
    if args.expected_manifest_sha256 is not None and manifest_sha256 != args.expected_manifest_sha256:
        raise ValueError('Manifest hash differs from --expected-manifest-sha256')
    if args.scenario == 'driving':
        if not args.bdd_detection or not args.bdd_mask_root:
            parser.error('--scenario driving requires --bdd-detection and --bdd-mask-root')
        bdd_gt = downstream_driving.ground_truth_loader(args.bdd_detection, args.bdd_mask_root)

    images, counts = _manifest(resolve_image_rows(read_jsonl(args.manifest), args.manifest, args.image_root), args.scenario, args.smoke_test)
    predictions = _unique(read_jsonl(args.predictions), ('image_id', 'model_id'))
    expected = {(image_id, model) for (image_id,), row in images.items() for model in MODELS[row['dataset']]}
    if set(predictions) != expected:
        raise ValueError(f'Prediction coverage mismatch: {len(expected - set(predictions))} missing, '
                         f'{len(set(predictions) - expected)} unexpected')
    for (image_id, model), entry in predictions.items():
        if entry['image_sha256'] != images[image_id,]['image_sha256']:
            raise ValueError(f'Prediction input hash differs for {image_id}, {model}')
    for (image_id,), row in images.items():
        if 'image' in row and file_sha256(_path(row['image'], args.manifest.resolve())) != row['image_sha256']:
            raise ValueError(f'Image bytes differ from the manifest for {image_id}')
    methods, quality = _composites(images, read_jsonl(args.quality_scores))

    scored = []
    for (image_id,), row in sorted(images.items()):
        dataset = row['dataset']
        if dataset == 'bdd100k':
            ground_truth = bdd_gt(row['source_image_name'])
        else:
            ground_truth = _path(row['ground_truth'], args.manifest.resolve())
        by_model = {}
        for model in MODELS[dataset]:
            entry = predictions[image_id, model]
            prediction = _path(entry['prediction'], args.predictions.resolve())
            if 'prediction_sha256' in entry and (
                not isinstance(prediction, Path) or file_sha256(prediction) != entry['prediction_sha256']
            ):
                raise ValueError(f'Prediction artifact hash differs for {image_id}, {model}')
            components = _task_score(dataset, prediction, ground_truth, model)
            if not 0 <= components['task_score'] <= 1:
                raise ValueError(f'Invalid task score for {image_id}, {model}')
            by_model[model] = components
        scored.append({
            'image_id': image_id, 'dataset': dataset, 'source_id': row['source_id'],
            'level': row['level'], 'models': by_model,
            'task_score': sum(values['task_score'] for values in by_model.values()) / 5,
        })
    report = {
        'scenario': args.scenario, 'smoke_test': args.smoke_test,
        'images': len(scored), 'sources_by_dataset': counts,
        'models_by_dataset': {dataset: MODELS[dataset] for dataset in counts},
        'input_sha256': {'manifest': manifest_sha256, 'predictions': file_sha256(args.predictions),
                         'quality_scores': file_sha256(args.quality_scores)},
        'correlations': _correlate(args.scenario, methods, quality, scored),
        'per_image': scored,
    }
    atomic_json(args.output, report)
    print(json.dumps({'images': len(scored), 'correlations': report['correlations']}, allow_nan=False))


if __name__ == '__main__':
    main()
