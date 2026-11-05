from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from .experiment_prompts import OFFICIAL_MOS_PROMPT
from ..paths import resolve_path


TABLE6_DATASETS = ("KonIQ-10k", "SPAQ", "KADID-10K", "PIPAL", "LIVEC", "AGIQA-3K", "CSIQ")
EXPECTED_COUNTS = dict(zip(TABLE6_DATASETS, (2015, 2225, 10125, 1000, 1162, 2982, 866)))
EXPECTED_COUNTS["TID2013"] = 3000
SCORE_FIELDS = {"KonIQ-10k": "mos", "SPAQ": "mos", "KADID-10K": "dmos",
                "PIPAL": "elo_score", "LIVEC": "mos", "AGIQA-3K": "mos_quality",
                "CSIQ": "dmos", "TID2013": "mos"}
SCORE_ANCHORS = list(range(0, 101, 10))
SCORE_DECODER = "expectation over eleven score-token probabilities"
ALIASES = {"koniq": "KonIQ-10k", "koniq10k": "KonIQ-10k", "koniq-10k": "KonIQ-10k",
           "spaq": "SPAQ", "kadid": "KADID-10K", "kadid10k": "KADID-10K",
           "kadid-10k": "KADID-10K", "pipal": "PIPAL", "livec": "LIVEC",
           "livew": "LIVEC", "live-wild": "LIVEC", "livechallenge": "LIVEC",
           "live-challenge": "LIVEC", "agiqa": "AGIQA-3K", "agiqa3k": "AGIQA-3K",
           "agiqa-3k": "AGIQA-3K", "csiq": "CSIQ", "tid2013": "TID2013"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_dataset(value: str) -> str:
    if not isinstance(value, str) or value.strip().lower() not in ALIASES:
        raise ValueError(f"Unsupported IQA dataset: {value!r}")
    return ALIASES[value.strip().lower()]


def _number(value, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _present(row, key):
    return key in row and row[key] is not None and row[key] != ""


def _input_rows(path: Path):
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            yield from csv.DictReader(handle)
    elif path.suffix.lower() in (".jsonl", ".ndjson"):
        with path.open(encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"Line {line_number}: expected an object")
                    yield row
    else:
        raise ValueError("IQA manifests must be external .csv or .jsonl files")


def load_iqa_cohort(path: str | Path, *, smoke_test: bool = False,
                    dataset: str | None = None, image_root: str | Path | None = None,
                    expected_datasets=None) -> list[dict]:

    path = Path(path).resolve()
    supplied_dataset = canonical_dataset(dataset) if dataset else None
    expected = tuple(canonical_dataset(x) for x in (expected_datasets or TABLE6_DATASETS))
    if len(set(expected)) != len(expected):
        raise ValueError("Expected dataset names must be unique")
    rows, ids, image_ids, image_cache = [], set(), set(), {}
    for index, original in enumerate(_input_rows(path), 1):
        row = dict(original)
        ds = canonical_dataset(row.get("dataset") or supplied_dataset)
        if supplied_dataset is not None and ds != supplied_dataset:
            raise ValueError(f"Row {index}: dataset disagrees with --dataset")
        split = str(row.get("official_split") or row.get("split") or row.get("set") or "").lower().strip()


        if ds == "KonIQ-10k" and split in ("train", "training", "val", "validation"):
            continue
        if ds == "PIPAL" and split in ("train", "training", "test", "testing"):
            continue
        image_value = (row.get("image") or row.get("image_path") or row.get("dist_name")
                       or row.get("name") or row.get("img_name") or row.get("image_name")
                       or row.get("dist_img") or row.get("filename"))
        if not isinstance(image_value, str) or not image_value.strip():
            raise ValueError(f"Row {index}: image/image_path or native filename required")
        image_name = Path(image_value).name.lower()
        if ds == "LIVEC" and (image_name in {f"t{x}.bmp" for x in range(1, 8)}
                              or "trainingimages" in image_value.lower().replace("\\", "/").split("/")):
            continue
        if not smoke_test:
            if ds == "KonIQ-10k" and split not in ("test", "testing", "official_test"):
                raise ValueError("KonIQ requires its official 2015-image test split")
            if ds == "PIPAL" and split not in ("val", "validation", "public_val"):
                raise ValueError("PIPAL requires the public labelled validation split, not hidden test")
            if ds == "SPAQ" and (split not in ("test", "holdout", "frozen_test")
                                  or not isinstance(row.get("cohort_id"), str)
                                  or not row["cohort_id"].strip()):
                raise ValueError("SPAQ requires a named frozen 2225-image holdout (split and cohort_id)")
        image = resolve_path(image_value, path, image_root)
        if not image.is_file():
            raise FileNotFoundError(f"Row {index}: image not found: {image}")
        if image not in image_cache:
            image_cache[image] = sha256_file(image)
        image_hash = image_cache[image]


        key = "eval_image_sha256" if _present(row, "eval_image_sha256") else "image_sha256"
        if _present(row, key) and str(row[key]).lower() != image_hash:
            raise ValueError(f"Row {index}: {key} does not match current image bytes")
        identifier = row.get("id") or f"{ds}:{image_value}"
        if not isinstance(identifier, str) or not identifier.strip() or identifier in ids:
            raise ValueError(f"Row {index}: id must be unique and nonempty")
        identity = (ds, str(image))
        if identity in image_ids:
            raise ValueError(f"Row {index}: duplicate image path within {ds}")
        ids.add(identifier)
        image_ids.add(identity)
        field = SCORE_FIELDS[ds]
        if _present(row, "score_field") and str(row["score_field"]).casefold() != field.casefold():
            raise ValueError(f"{ds} uses {field}, not {row['score_field']!r}")
        direction = -1 if ds == "CSIQ" else 1
        if _present(row, "direction_multiplier") and _number(row["direction_multiplier"], "direction_multiplier") != direction:
            raise ValueError(f"{ds}: direction_multiplier must be {direction}")
        sources = {}
        if _present(row, "raw_score"):
            sources["raw_score"] = _number(row["raw_score"], "raw_score")
        for key in row:
            if key.casefold() == field.casefold() and _present(row, key):
                sources[key] = _number(row[key], key)
        if not sources and _present(row, "target"):
            if not _present(row, "score_field") or not _present(row, "direction_multiplier"):
                raise ValueError("Target-only rows must declare score_field and direction_multiplier")
            sources["directed_target"] = _number(row["target"], "target") / direction
        if not sources:
            raise ValueError(f"Row {index}: raw_score or native {field} column required")
        if len(set(sources.values())) != 1:
            raise ValueError(f"Row {index}: conflicting raw/native score fields: {sources}")
        raw = next(iter(sources.values()))
        target = raw * direction
        if _present(row, "target") and _number(row["target"], "target") != target:
            raise ValueError(f"Row {index}: target disagrees with directed raw score")
        rows.append({"id": identifier, "image": str(image), "dataset": ds,
                     "raw_score": raw, "target": target, "score_field": field,
                     "direction_multiplier": direction, "image_sha256": image_hash,
                     "split": split, "cohort_id": row.get("cohort_id", "")})
    validate_iqa_cohort(rows, smoke_test=smoke_test, expected_datasets=expected)
    return rows


def validate_iqa_cohort(rows, *, smoke_test=False, expected_datasets=TABLE6_DATASETS):
    if not rows:
        raise ValueError("The IQA cohort is empty")
    counts = {}
    for row in rows:
        ds = row["dataset"]
        counts[ds] = counts.get(ds, 0) + 1
    if not smoke_test:
        expected = tuple(canonical_dataset(x) for x in expected_datasets)
        if set(counts) != set(expected):
            raise ValueError(f"Expected datasets {sorted(expected)}, received {sorted(counts)}")
        for ds in expected:
            if counts[ds] != EXPECTED_COUNTS[ds]:
                raise ValueError(f"{ds}: expected {EXPECTED_COUNTS[ds]} images, received {counts[ds]}")
        if "SPAQ" in counts and len({r["cohort_id"] for r in rows if r["dataset"] == "SPAQ"}) != 1:
            raise ValueError("All SPAQ rows must belong to one frozen holdout cohort_id")


def cohort_sha256(rows) -> str:

    fields = ("id", "dataset", "image_sha256", "raw_score", "target", "score_field",
              "direction_multiplier", "split", "cohort_id")
    payload = [{key: row.get(key, "") for key in fields} for row in sorted(rows, key=lambda r: r["id"])]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def build_iqa_requests(rows) -> list[dict]:
    return [{"id": row["id"], "image": row["image"], "prompt": OFFICIAL_MOS_PROMPT,
             "long_edge": 448} for row in rows]


def requests_sha256(rows) -> str:
    records = [{"id": row["id"], "image_sha256": row["image_sha256"],
                "long_edge": 448, "prompt": OFFICIAL_MOS_PROMPT} for row in rows]
    payload = json.dumps(sorted(records, key=lambda r: r["id"]), sort_keys=True,
                         separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def evaluate_iqa(rows, predictions, *, smoke_test=False, expected_datasets=TABLE6_DATASETS,
                 prediction_cohort_sha256=None) -> dict:

    validate_iqa_cohort(rows, smoke_test=smoke_test, expected_datasets=expected_datasets)
    fingerprint = cohort_sha256(rows)
    if prediction_cohort_sha256 is not None and prediction_cohort_sha256 != fingerprint:
        raise ValueError("Prediction receipt belongs to a different IQA image/label cohort")
    truth = {row["id"]: row for row in rows}
    if len(truth) != len(rows):
        raise ValueError("Duplicate IDs in IQA cohort")
    mapped = {}
    for row in predictions:
        if not isinstance(row, dict):
            raise ValueError("Each IQA prediction must be an object")
        if "valid" in row and (type(row["valid"]) is not bool or row["valid"] is not True):
            raise ValueError("Predictions explicitly marked invalid cannot be excluded or imputed")
        if row.get("parse_status", "valid") not in ("valid", "ok"):
            raise ValueError("Invalid parsed predictions cannot be excluded or imputed")
        identifier = row.get("id")
        if identifier in mapped or identifier not in truth:
            raise ValueError(f"Duplicate or unexpected prediction id: {identifier!r}")
        if ("score" in row) == ("prediction" in row):
            raise ValueError("Each prediction requires exactly one of score or prediction")
        value = _number(row.get("score", row.get("prediction")), "prediction")
        if not 0 <= value <= 100:
            raise ValueError("Model predictions must lie in [0, 100]; no clipping is applied")
        if "target" in row and _number(row["target"], "prediction target") != truth[identifier]["target"]:
            raise ValueError(f"Prediction target disagrees with the frozen label: {identifier}")
        if "image_sha256" in row and row["image_sha256"] != truth[identifier]["image_sha256"]:
            raise ValueError(f"Prediction image hash disagrees with the frozen image: {identifier}")
        mapped[identifier] = value
    if set(mapped) != set(truth):
        raise ValueError(f"Incomplete predictions: missing {len(set(truth) - set(mapped))} IQA IDs")
    from .common import correlations
    results = {}
    for ds in sorted({row["dataset"] for row in rows}):
        selected = [r for r in rows if r["dataset"] == ds]
        targets = [r["target"] for r in selected]
        scores = [mapped[r["id"]] for r in selected]
        metrics = correlations(targets, scores)
        status = "complete" if metrics["undefined_reason"] is None else "undefined_constant_or_insufficient_values"
        results[ds] = {**metrics, "status": status, "score_field": SCORE_FIELDS[ds],
                       "direction_multiplier": -1 if ds == "CSIQ" else 1}
    return {"format": "oasis_iqa_evaluation_v1", "benchmark": not smoke_test,
            "smoke_test": smoke_test,
            "suite": "Table6" if not smoke_test and set(results) == set(TABLE6_DATASETS)
                     else "smoke_fixture" if smoke_test else "explicit_dataset_subset",
            "cohort_sha256": fingerprint, "n": len(rows), "datasets": results,
            "prompt_sha256": hashlib.sha256(OFFICIAL_MOS_PROMPT.encode()).hexdigest(),
            "protocol": "raw PLCC/SRCC; no test-fitted mapping, clipping, imputation or model-wise cohort filtering",
            "error_metrics": None,
            "error_note": "Raw label scales differ across datasets; MAE/RMSE are not compared."}


def _read_predictions(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _validate_decoder(receipt):
    if receipt.get("decoder") != SCORE_DECODER or receipt.get("score_anchors") != SCORE_ANCHORS:
        raise ValueError("IQA comparisons require the eleven-anchor expectation decoder over 0,10,...,100")


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run", "evaluate"):
        command = sub.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--dataset")
        command.add_argument("--datasets", nargs="+", default=list(TABLE6_DATASETS))
        command.add_argument("--image-root", type=Path)
        command.add_argument("--smoke-test", action="store_true")
        command.add_argument("--output-dir", type=Path, required=True)
    runner = sub.choices["run"]
    runner.add_argument("--model-dir", type=Path, required=True)
    runner.add_argument("--state-file", type=Path)
    runner.add_argument("--codebook", type=Path, required=True)
    runner.add_argument("--device", default="cuda:0")
    runner.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    runner.add_argument("--image-policy", choices=("paper", "archive-jpeg95"), default="paper")
    runner.add_argument("--already-prepared", action="store_true")
    sub.choices["evaluate"].add_argument("--run", action="append", required=True,
                                       metavar="MODEL=DIR")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    rows = load_iqa_cohort(args.manifest, smoke_test=args.smoke_test, dataset=args.dataset,
                           image_root=args.image_root, expected_datasets=args.datasets)
    fingerprint = cohort_sha256(rows)
    payload = {"format": "oasis_iqa_cohort_v1", "cohort_sha256": fingerprint,
               "source_manifest_sha256": sha256_file(args.manifest), "n": len(rows),
               "smoke_test": args.smoke_test, "rows": rows}
    reports = {}
    if args.command == "evaluate":
        policies = set()
        for spec in args.run:
            if "=" not in spec:
                raise ValueError("--run must be MODEL=DIR")
            name, value = spec.split("=", 1)
            if not name or name in reports:
                raise ValueError("Prediction model names must be nonempty and unique")
            directory = Path(value)
            path = directory / "predictions.jsonl"
            receipt_path = directory / "run.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("format") != "oasis_experiment_run_v1":
                raise ValueError(f"Expected a successful core experiment receipt: {receipt_path}")
            _validate_decoder(receipt)
            if receipt.get("cohort_sha256") != fingerprint:
                raise ValueError(f"Missing or mismatched cohort receipt: {receipt_path}")
            if receipt.get("predictions_sha256") != sha256_file(path):
                raise ValueError(f"Prediction bytes differ from the successful receipt: {path}")
            if receipt.get("requests_sha256") != requests_sha256(rows) or receipt.get("image_only") is not False:
                raise ValueError("IQA runs must use the identical full official MOS prompt and frozen image inputs")
            if receipt.get("requests") != len(rows):
                raise ValueError("IQA run receipt does not cover the full supplied cohort")


            policies.add(json.dumps({key: receipt.get(key) for key in
                                     ("image_policy", "already_prepared", "dtype")},
                                    sort_keys=True))
            reports[name] = evaluate_iqa(rows, _read_predictions(path), smoke_test=args.smoke_test,
                                         expected_datasets=args.datasets,
                                         prediction_cohort_sha256=receipt.get("cohort_sha256"))
            reports[name]["run"] = {key: receipt.get(key) for key in
                                     ("checkpoint_sha256", "codebook_sha256", "requests_sha256",
                                      "image_policy", "already_prepared", "dtype", "decoder", "score_anchors")}
        if len(policies) != 1:
            raise ValueError("IQA model comparisons must use the same image policy, preparation and dtype")
    elif args.command == "run":
        from .common import score_requests
        scored = score_requests(build_iqa_requests(rows), model_dir=args.model_dir,
                                codebook=args.codebook, output_dir=args.output_dir,
                                state_file=args.state_file, device=args.device, dtype=args.dtype,
                                image_policy=args.image_policy, already_prepared=args.already_prepared,
                                cohort_sha256=fingerprint)
        _validate_decoder(scored["manifest"])
        if scored["manifest"].get("cohort_sha256") != fingerprint:
            raise ValueError("The scorer did not bind its predictions to the frozen IQA cohort")
        if scored["manifest"].get("requests_sha256") != requests_sha256(rows):
            raise ValueError("Image bytes or IQA request content changed after the cohort was frozen")
        report = evaluate_iqa(rows, scored["predictions"], smoke_test=args.smoke_test,
                              expected_datasets=args.datasets,
                              prediction_cohort_sha256=scored["manifest"].get("cohort_sha256"))
        _write_json(args.output_dir / "cohort.json", payload)
        _write_json(args.output_dir / "metrics.json", report)
        return
    args.output_dir.mkdir(parents=True)
    _write_json(args.output_dir / "cohort.json", payload)
    if args.command == "prepare":
        with (args.output_dir / "requests.jsonl").open("x", encoding="utf-8") as handle:
            for row in build_iqa_requests(rows):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        _write_json(args.output_dir / "metrics.json", {"format": "oasis_iqa_comparison_v1",
                    "cohort_sha256": fingerprint, "smoke_test": args.smoke_test, "models": reports})


if __name__ == "__main__":
    main()
