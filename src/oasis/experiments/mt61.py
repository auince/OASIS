from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Iterable

from .experiment_prompts import VARIANTS, build_variant_prompt, extract_definition

SCENE_DIMENSIONS = {"autonomous_driving": 6, "depth_estimation": 5, "medical": 5, "remote_sensing": 5}
SCENE_ALIASES = {"medical_imaging": "medical", "autonomous driving": "autonomous_driving",
                 "depth estimation": "depth_estimation", "remote sensing": "remote_sensing"}


def normalize_rows(records: Iterable[dict]) -> list[dict]:

    rows, seen, dimension_scenes = [], set(), {}
    for index, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index}: expected an object")
        row = dict(record)
        for key in ("id", "image", "scene"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Record {index}: {key} must be a nonempty string")
        if row["id"] in seen:
            raise ValueError(f"Duplicate reference id: {row['id']}")
        seen.add(row["id"])
        dimension = row.get("dimension", row.get("task"))
        if not isinstance(dimension, str) or not dimension.strip():
            raise ValueError(f"Record {index}: dimension or task must be a nonempty string")
        if "dimension" in row and "task" in row and row["dimension"] != row["task"]:
            raise ValueError(f"Record {index}: dimension and task disagree")
        row["scene"] = SCENE_ALIASES.get(row["scene"], row["scene"])
        if row["scene"] not in SCENE_DIMENSIONS:
            raise ValueError(f"Record {index}: unknown MT61 scene {row['scene']!r}")
        if dimension in dimension_scenes and dimension_scenes[dimension] != row["scene"]:
            raise ValueError(f"Dimension {dimension!r} occurs in multiple scenes")
        dimension_scenes[dimension] = row["scene"]
        score = row.get("score", row.get("target"))
        if "target" in row:
            target = row["target"]
            if isinstance(target, bool) or not isinstance(target, (int, float)) or not math.isfinite(target):
                raise ValueError(f"Record {index}: target must be finite numeric data")
            if "score" in row and row["score"] != target:
                raise ValueError(f"Record {index}: score and target disagree")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 100:
            raise ValueError(f"Record {index}: score must be finite and in [0, 100]")
        edge = row.get("long_edge", 720 if row["scene"] == "remote_sensing" else 448)
        if type(edge) is not int or edge not in (448, 720):
            raise ValueError(f"Record {index}: long_edge must be 448 or 720")
        row.update(dimension=dimension, long_edge=edge, score=score, target=score)
        if "image_sha256" not in row and "eval_image_sha256" in row:
            row["image_sha256"] = row["eval_image_sha256"]
        rows.append(row)
    if not rows:
        raise ValueError("No reference records supplied")
    return rows


def _image_identity(row: dict) -> str:
    if isinstance(row.get("image_id"), str) and row["image_id"]:
        return row["image_id"]
    suffix = ":" + row["dimension"]
    return row["id"][:-len(suffix)] if row["id"].endswith(suffix) else row["image"]


def validate_dimensions(rows: list[dict], *, smoke_test: bool = False) -> dict:
    groups = {scene: {row["dimension"] for row in rows if row["scene"] == scene}
              for scene in SCENE_DIMENSIONS}
    counts = {scene: len(values) for scene, values in groups.items()}
    image_count = len({_image_identity(row) for row in rows})
    complete = counts == SCENE_DIMENSIONS and len(rows) == 33670 and image_count == 6492
    if not complete and not smoke_test:
        raise ValueError(f"A full MT61-Open evaluation requires dimension counts {SCENE_DIMENSIONS}, "
                         f"33670 records and 6492 images; received {counts}, {len(rows)} records "
                         f"and {image_count} images. Use --smoke-test only for a nonbenchmark subset.")
    return {"complete_21_dimension_benchmark": complete, "dimension_counts": counts,
            "records": len(rows), "images": image_count, "smoke_test": smoke_test,
            "benchmark_result": complete and not smoke_test}


def build_requests(rows: list[dict], variant: str) -> list[dict]:

    requests = []
    for row in rows:
        if variant in ("full_prompt", "definition_only", "name_only"):
            if extract_definition(row)["name"].strip() != row["dimension"].strip():
                raise ValueError(f"Criterion name does not match the reference dimension for {row['id']!r}")
        requests.append({"id": row["id"], "image": row["image"], "long_edge": row["long_edge"],
                         "prompt": build_variant_prompt(row, variant)})
    return requests


def _macro(groups: dict, n: int) -> dict:
    undefined = [name for name, value in groups.items()
                 if value["plcc"] is None or value["srcc"] is None]
    if not groups:
        reason = "No eligible dimensions"
    elif undefined:
        reason = "Undefined correlation in one or more dimensions; none were silently omitted"
    else:
        reason = None
    return {"n": n, "dimensions": len(groups),
            "plcc": None if reason else sum(value["plcc"] for value in groups.values()) / len(groups),
            "srcc": None if reason else sum(value["srcc"] for value in groups.values()) / len(groups),
            "undefined_reason": reason, "undefined_dimensions": undefined}


def metrics_from_scores(rows: list[dict], scores: list[float], *, smoke_test: bool = False) -> dict:

    from .common import correlations
    coverage = validate_dimensions(rows, smoke_test=smoke_test)
    if len(rows) != len(scores):
        raise ValueError("A prediction is required for every reference record")
    grouped = defaultdict(list)
    for row, prediction in zip(rows, scores):
        if isinstance(prediction, bool) or not isinstance(prediction, (int, float)) or not math.isfinite(prediction):
            raise ValueError(f"Prediction must be finite for id {row['id']!r}")
        if not 0 <= prediction <= 100:
            raise ValueError(f"Prediction outside [0, 100] for id {row['id']!r}")
        grouped[row["dimension"]].append((row["score"], prediction, row["scene"]))
    per_dimension = {}
    for dimension, pairs in sorted(grouped.items()):
        per_dimension[dimension] = dict(correlations([x[0] for x in pairs], [x[1] for x in pairs]),
                                        scene=pairs[0][2])
    by_scene = {}
    for scene in SCENE_DIMENSIONS:
        scene_groups = {name: value for name, value in per_dimension.items() if value["scene"] == scene}
        by_scene[scene] = _macro(scene_groups, sum(row["scene"] == scene for row in rows))
    return {"protocol": "dimension_macro_raw", "coverage": coverage,
            "overall": _macro(per_dimension, len(rows)), "by_scene": by_scene,
            "per_dimension": per_dimension,
            "remote_sensing_protocol": "Pool applicable iSAID and SpaceNet3 records within each dimension, "
                                       "compute each dimension correlation, then average dimensions equally",
            "score_mapping": "none; raw PLCC and SRCC; no test-set fitted transform"}


def evaluate_rows(records: Iterable[dict], predictions: Iterable[dict], *, smoke_test: bool = False) -> dict:
    from .common import cohort_fingerprint, join_predictions
    rows = normalize_rows(records)
    joined = join_predictions(rows, list(predictions))
    result = metrics_from_scores(joined, [row["prediction"] for row in joined], smoke_test=smoke_test)
    result["cohort_sha256"] = cohort_fingerprint(rows)
    return result


def validate_run_receipt(path: Path, predictions_path: Path, fingerprint: str, variant: str) -> dict:
    from .common import file_sha256
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("cohort_sha256") != fingerprint:
        raise ValueError("Prediction run manifest cohort_sha256 does not match the reference cohort")
    if manifest.get("predictions_sha256") != file_sha256(predictions_path):
        raise ValueError("Prediction file bytes do not match the inference receipt predictions_sha256")
    source_protocol_path = path.with_name("protocol.json")
    if not source_protocol_path.is_file():
        raise ValueError("A source protocol.json beside run.json is required to verify the input variant")
    source_protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
    if source_protocol.get("variant") != variant or source_protocol.get("cohort_sha256") != fingerprint:
        raise ValueError("Source experiment protocol variant or cohort does not match this evaluation")
    recorded_run = source_protocol.get("run_manifest", {})
    if recorded_run != manifest:
        raise ValueError("Source protocol is not bound to the supplied inference run receipt")
    if manifest.get("image_policy") not in ("paper", "archive-jpeg95"):
        raise ValueError("Inference receipt must identify its explicit image preparation policy")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "evaluate"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--input", type=Path, required=True)
        sub.add_argument("--image-root", type=Path)
        sub.add_argument("--output-dir", type=Path, required=True)
        sub.add_argument("--variant", choices=VARIANTS, required=True)
        sub.add_argument("--smoke-test", action="store_true")
        sub.add_argument("--expected-cohort-sha256")
        if command == "run":
            sub.add_argument("--model-dir", type=Path, required=True)
            sub.add_argument("--state-file", type=Path)
            sub.add_argument("--codebook", type=Path, required=True)
            sub.add_argument("--device", default="cuda:0")
            sub.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
            sub.add_argument("--image-policy", choices=("paper", "archive-jpeg95"), default="paper")
            sub.add_argument("--already-prepared", action="store_true")
        else:
            sub.add_argument("--predictions", type=Path, required=True)
            sub.add_argument("--run-manifest", type=Path)
    args = parser.parse_args(argv)
    from .common import (atomic_json, cohort_fingerprint, file_sha256, join_predictions,
                         read_jsonl, score_requests, resolve_image_rows)
    rows = normalize_rows(resolve_image_rows(read_jsonl(args.input), args.input, args.image_root))
    coverage = validate_dimensions(rows, smoke_test=args.smoke_test)
    fingerprint = cohort_fingerprint(rows)
    if args.expected_cohort_sha256 and args.expected_cohort_sha256 != fingerprint:
        raise ValueError("Reference cohort differs from --expected-cohort-sha256")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    protocol = {"experiment": "mt61", "variant": args.variant,
                "cohort_sha256": fingerprint, "reference_file_sha256": file_sha256(args.input),
                "coverage": coverage,
                "reference_provenance": "As supplied in the external reference records; no annotation source is inferred",
                "aggregation": "Correlate within each dimension, then average all 21 dimensions equally; "
                               "scene reports average their 6/5/5/5 dimensions; remote sources are pooled within dimensions",
                "correlation_units": "image-dimension reference records"}
    if args.command == "run":
        requests = build_requests(rows, args.variant)
        scored = score_requests(requests, model_dir=args.model_dir, codebook=args.codebook,
                                output_dir=args.output_dir, state_file=args.state_file,
                                device=args.device, dtype=args.dtype, image_policy=args.image_policy,
                                already_prepared=args.already_prepared, cohort_sha256=fingerprint)
        predictions = scored["predictions"]
        protocol.update(image_policy=args.image_policy, already_prepared=args.already_prepared,
                        prediction_provenance="Generated by this run", variant_verified=True,
                        run_manifest=scored["manifest"])
    else:
        predictions = read_jsonl(args.predictions)
        protocol.update(prediction_file_sha256=file_sha256(args.predictions),
                        prediction_provenance="External predictions; prompt and image policy are not inferred",
                        variant_verified=False)
        if args.run_manifest:
            manifest = validate_run_receipt(args.run_manifest, args.predictions, fingerprint, args.variant)
            protocol.update(run_manifest=manifest, variant_verified=True,
                            image_policy=manifest["image_policy"],
                            already_prepared=manifest.get("already_prepared"),
                            run_manifest_sha256=file_sha256(args.run_manifest),
                            source_protocol_sha256=file_sha256(args.run_manifest.with_name("protocol.json")))

        join_predictions(rows, predictions)
        args.output_dir.mkdir(parents=True, exist_ok=False)
    result = evaluate_rows(rows, predictions, smoke_test=args.smoke_test)
    if result["cohort_sha256"] != fingerprint:
        raise ValueError("Reference image bytes or cohort changed after this experiment was frozen")
    result["variant"] = args.variant
    atomic_json(args.output_dir / "protocol.json", protocol)
    atomic_json(args.output_dir / "metrics.json", result)
    print(json.dumps({"output_dir": str(args.output_dir), "variant": args.variant,
                      "overall": result["overall"], "coverage": coverage}, allow_nan=False))


if __name__ == "__main__":
    main()
