import argparse
import json
import math
import os
from pathlib import Path
import tempfile

from .paths import resolve_path, relative_path

FIELDS = ('name', 'description', 'positive_evidence', 'negative_evidence', 'score_guidance')
CAPTION_INSTRUCTION = (
    'Analyze the image as a low-level and task-oriented image-quality expert. '
    'Use every applicable dimension definition below. For each dimension, explain '
    'concrete visible strengths and limitations, weighting severity, affected area, '
    'and task relevance. Do not output a numeric score, score range, dataset name, '
    'or a final token. Keep the analysis evidence-based.'
)


def convert(input_path, dimensions_path, stage, image_root=None, output_path=None):
    input_path = Path(input_path).resolve()
    dimensions = json.loads(Path(dimensions_path).read_text(encoding='utf-8'))
    if not isinstance(dimensions, dict) or not dimensions:
        raise ValueError('Dimensions must be a nonempty object keyed by stable dimension identifiers')
    if stage not in ('caption', 'score', 'mt61'):
        raise ValueError('Invalid stage')
    result, seen = [], set()
    for number, line in enumerate(input_path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f'Line {number}: expected an object')
        if any(not isinstance(row.get(k), str) or not row[k].strip()
               for k in ('image', 'dimension', 'description')):
            raise ValueError(f'Line {number}: nonempty image, dimension, description required')
        image_key, dimension = row['image'], row['dimension']
        if Path(image_key).is_absolute():
            raise ValueError(f'Line {number}: image must be relative to the input JSONL')
        image = resolve_path(image_key, input_path, root=image_root)
        if not image.is_file():
            raise FileNotFoundError(image)
        identifier = image_key + ':' + dimension
        if identifier in seen:
            raise ValueError(f'Line {number}: duplicate image/dimension pair')
        seen.add(identifier)
        definition = dimensions.get(dimension)
        if not isinstance(definition, dict) or any(
            not isinstance(definition.get(k), str) or not definition[k].strip() for k in FIELDS
        ):
            raise ValueError(f'Line {number}: incomplete dimension definition for {dimension}')
        if definition.get('role') not in ('base', 'open'):
            raise ValueError(f'Line {number}: dimension role must be base or open')
        score = row.get('score')
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 100:
            raise ValueError(f'Line {number}: score must be finite and in [0, 100]')
        parts = Path(image_key).parts
        if (len(parts) != 4 or parts[0] != 'images' or parts[1] not in ('base', 'open')
                or parts[2] not in ('448', '720') or Path(parts[3]).suffix.lower() != '.jpg'):
            raise ValueError(f'Line {number}: expected images/base|open/448|720/filename.jpg')
        if definition['role'] != parts[1]:
            raise ValueError(f'Line {number}: image role and dimension role disagree')
        if stage in ('caption', 'score') and definition['role'] != 'base':
            raise ValueError('Training requires base dimensions')
        edge = int(parts[2])
        structured = {k: definition[k] for k in FIELDS}
        converted = dict(id=identifier, image=relative_path(image, output_path) if output_path is not None else str(image), image_id=image_key,
                         dimension=dimension, score=score, long_edge=edge)
        if stage == 'caption':
            converted['prompt'] = (
                CAPTION_INSTRUCTION + '\n\n[' + structured['name'] + ']\n'
                + 'Definition: ' + structured['description'] + '\n'
                + 'Positive evidence: ' + structured['positive_evidence'] + '\n'
                + 'Negative evidence: ' + structured['negative_evidence'])
            converted['answer'] = row['description']
        else:
            converted['definition'] = structured
        if stage == 'mt61':
            if definition['role'] != 'open':
                raise ValueError('MT61 evaluation requires open dimensions')
            scene = definition.get('scene')
            if scene not in ('autonomous_driving', 'depth_estimation', 'medical', 'remote_sensing'):
                raise ValueError(f'Line {number}: missing or invalid open-dimension scene')
            if structured['name'] != dimension:
                raise ValueError('MT61 requires definition.name to match the stable dimension identifier')
            converted['scene'] = scene
        result.append(converted)
    if not result:
        raise ValueError('Input has no records')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--dimensions', type=Path, required=True)
    parser.add_argument('--image-root', type=Path)
    parser.add_argument('--stage', choices=('caption', 'score', 'mt61'), required=True)
    parser.add_argument('--already-prepared', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    records = convert(args.input, args.dimensions, args.stage,
                      image_root=args.image_root, output_path=args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=args.output.parent,
                                         prefix='.' + args.output.name + '.', delete=False) as stream:
            temporary = Path(stream.name)
            for row in records:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, args.output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    summary = {'stage': args.stage, 'records': len(records),
               'already_prepared': args.already_prepared,
               'training_image_flag': '--already-prepared' if args.already_prepared else None}
    if args.stage == 'score':
        summary['augmentation_requirement'] = (
            'Pass --no-definition-augmentation to training, or supply externally reviewed paraphrases')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
