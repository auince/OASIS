from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import tempfile

import numpy as np

from ..paths import resolve_path

_DIGEST_CACHE = {}


def resolve_image_rows(rows, reference_file, image_root=None, fields=('image',)):
    result = []
    for original in rows:
        row = dict(original)
        for field in fields:
            if field in row:
                row[field] = str(resolve_path(row[field], reference_file, image_root))
        result.append(row)
    return result


def file_sha256(path):
    path = Path(path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _DIGEST_CACHE:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        _DIGEST_CACHE[key] = digest.hexdigest()
    return _DIGEST_CACHE[key]


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f'{path}:{number}: invalid JSON') from error
            if not isinstance(row, dict):
                raise ValueError(f'{path}:{number}: expected a JSON object')
            rows.append(row)
    if not rows:
        raise ValueError(f'{path}: no records')
    return rows


def atomic_json(path, payload):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=f'.{path.name}.', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _finite(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f'{label} must be a finite number')
    return float(value)


def _rank_average(values):
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values), dtype=np.float64)
    begin = 0
    while begin < len(values):
        end = begin + 1
        while end < len(values) and values[order[end]] == values[order[begin]]:
            end += 1
        ranks[order[begin:end]] = (begin + end - 1) / 2 + 1
        begin = end
    return ranks


def _pearson(x, y):
    x, y = x - x.mean(), y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    if denominator == 0:
        return None
    value = float(np.dot(x, y) / denominator)
    if not math.isfinite(value):
        raise ValueError('Numerical overflow in correlation calculation')
    return float(np.clip(value, -1.0, 1.0))


def correlations(targets, predictions):

    targets, predictions = list(targets), list(predictions)
    if len(targets) != len(predictions):
        raise ValueError('Reference and prediction counts differ')
    x = np.asarray([_finite(v, 'reference') for v in targets], dtype=np.float64)
    y = np.asarray([_finite(v, 'prediction') for v in predictions], dtype=np.float64)
    reason = None
    if len(x) < 2:
        reason = 'fewer_than_two_records'
    elif np.ptp(x) == 0 or np.ptp(y) == 0:
        reason = 'constant_reference_or_prediction'
    if reason:
        return {'n': len(x), 'plcc': None, 'srcc': None, 'undefined_reason': reason}
    return {'n': len(x), 'plcc': _pearson(x, y),
            'srcc': _pearson(_rank_average(x), _rank_average(y)), 'undefined_reason': None}


def _unique(rows, label):
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('id'), str) or not row['id'].strip():
            raise ValueError(f'{label} records need nonempty string IDs')
        if row['id'] in result:
            raise ValueError(f'Duplicate {label} ID: {row["id"]}')
        result[row['id']] = row
    if not result:
        raise ValueError(f'No {label} records')
    return result


def join_predictions(expected_rows, predictions):

    expected_rows = list(expected_rows)
    expected, found = _unique(expected_rows, 'reference'), _unique(predictions, 'prediction')
    missing, extra = sorted(expected.keys() - found.keys()), sorted(found.keys() - expected.keys())
    if missing or extra:
        raise ValueError(f'Prediction cohort differs: {len(missing)} missing, {len(extra)} extra; '
                         f'examples missing={missing[:3]}, extra={extra[:3]}')
    joined = []
    for gold in expected_rows:
        row = found[gold['id']]
        if 'valid' in row and row['valid'] is not True:
            raise ValueError(f'Explicitly invalid prediction: {gold["id"]}')
        if row.get('parse_status', 'valid') != 'valid':
            raise ValueError(f'Invalid prediction status: {gold["id"]}')
        prediction = _finite(row.get('prediction', row.get('score')), f'prediction {gold["id"]}')
        if 'prediction' in row and 'score' in row and row['prediction'] != row['score']:
            raise ValueError(f'Ambiguous prediction/score fields: {gold["id"]}')
        target = _finite(gold.get('target'), f'reference {gold["id"]}')
        if 'target' in row and _finite(row['target'], 'embedded target') != target:
            raise ValueError(f'Embedded reference differs from the supplied cohort: {gold["id"]}')
        joined.append({**gold, 'prediction': prediction})
    return joined


def _image_digest(row):
    path = Path(row['image'])
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = file_sha256(path)


    declared = row.get('eval_image_sha256')
    if declared is not None and declared != actual:
        raise ValueError(f'Input image hash mismatch: {row.get("id", str(path))}')
    return actual


def cohort_fingerprint(rows):

    rows = list(rows)
    _unique(rows, 'cohort')
    records = []
    for row in rows:
        records.append({
            'id': row['id'], 'dataset': row.get('dataset'),
            'dimension': row.get('dimension', row.get('task')),
            'scene': row.get('scene'), 'target': _finite(row.get('target'), 'reference'),
            'image_sha256': _image_digest(row),
            'split': row.get('split'), 'cohort_id': row.get('cohort_id'),
            'score_source': row.get('score_source'), 'reference_source': row.get('reference_source'),
        })
    payload = json.dumps(sorted(records, key=lambda r: r['id']), sort_keys=True,
                         separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def score_requests(requests, *, model_dir, codebook, output_dir, state_file=None,
                   device='cuda:0', dtype='bfloat16', image_policy='paper',
                   already_prepared=False, cohort_sha256=None):

    from ..inference import main as score_main, read_codebook
    from ..prompts import build_prompt

    requests = list(requests)
    _unique(requests, 'request')
    if image_policy not in ('paper', 'archive-jpeg95'):
        raise ValueError('Invalid image policy')
    model_dir, codebook, output_dir = Path(model_dir), Path(codebook), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite {output_dir}')
    state_file = Path(state_file) if state_file is not None else model_dir / 'full_train_state.pt'
    read_codebook(codebook)
    if not (model_dir / 'config.json').is_file() or not state_file.is_file():
        raise FileNotFoundError('Provide a complete merged model and its full state')
    unique, mapping, identity_records = {}, {}, []
    for row in requests:
        has_prompt, has_definition = 'prompt' in row, 'definition' in row
        if has_prompt == has_definition:
            raise ValueError('Each scoring request needs exactly one prompt or definition')
        prompt = row['prompt'] if has_prompt else build_prompt(row['definition'])
        if not isinstance(prompt, str) or (prompt != '' and not prompt.strip()):
            raise ValueError('Prompt must be nonempty text or exactly empty for image-only')
        edge = row.get('long_edge', 448)
        if type(edge) is not int or edge not in (448, 720):
            raise ValueError('long_edge must be 448 or 720')
        image_digest = _image_digest(row)
        key = (image_digest, edge, prompt)
        if key not in unique:
            unique[key] = {'id': row['id'], 'image': str(Path(row['image']).resolve()),
                           'long_edge': edge, 'prompt': prompt}
        mapping[row['id']] = unique[key]['id']
        identity_records.append({'id': row['id'], 'image_sha256': image_digest,
                                 'long_edge': edge, 'prompt': prompt})
    image_only = {r['prompt'] == '' for r in unique.values()}
    if len(image_only) != 1:
        raise ValueError('Run image-only and criterion-conditioned requests in separate runs')
    input_bytes = json.dumps(sorted(identity_records, key=lambda r: r['id']), sort_keys=True,
                             separators=(',', ':'), ensure_ascii=False).encode()
    manifest = {
        'format': 'oasis_experiment_run_v1', 'cohort_sha256': cohort_sha256,
        'checkpoint_sha256': file_sha256(state_file), 'codebook_sha256': file_sha256(codebook),
        'model_config_sha256': file_sha256(model_dir / 'config.json'),
        'requests_sha256': hashlib.sha256(input_bytes).hexdigest(),
        'image_policy': image_policy, 'already_prepared': bool(already_prepared),
        'dtype': dtype, 'image_only': True in image_only,
        'requests': len(requests), 'unique_model_inputs': len(unique),
        'decoder': 'expectation over eleven score-token probabilities',
        'score_anchors': list(range(0, 101, 10)),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        with tempfile.TemporaryDirectory(prefix='oasis-scoring-') as directory:
            temporary = Path(directory)
            input_path = temporary / 'requests.jsonl'
            input_path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n'
                                          for r in unique.values()), encoding='utf-8')
            output_path = temporary / 'predictions.jsonl'
            argv = ['--model-dir', str(model_dir), '--state-file', str(state_file),
                    '--codebook', str(codebook), '--input', str(input_path), '--output', str(output_path),
                    '--device', device, '--dtype', dtype, '--image-policy', image_policy]
            if True in image_only:
                argv.append('--image-only')
            if already_prepared:
                argv.append('--already-prepared')
            score_main(argv)
            scored = _unique(read_jsonl(output_path), 'scored')
        if set(scored) != {value['id'] for value in unique.values()}:
            raise RuntimeError('Core scorer returned an incomplete request set')
        expanded = [{**scored[mapping[row['id']]], 'id': row['id']} for row in requests]
        destination = output_dir / 'predictions.jsonl'
        with destination.open('x', encoding='utf-8') as stream:
            for row in expanded:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')

        manifest['predictions_sha256'] = file_sha256(destination)
        atomic_json(output_dir / 'run.json', manifest)
        return {'predictions': expanded, 'manifest': manifest}
    except BaseException as error:
        atomic_json(output_dir / 'failure.json', {'status': 'failed', 'error': str(error)})
        raise
