from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib import metadata as package_metadata
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

from ..augmentation import validate_augmentation_record, validate_definition
from ..paths import resolve_path


SETTINGS = {"K": {"K"}, "K+S": {"K", "S"}, "K+S+M": {"K", "S", "M"}}
DATASET_NAMES = {"k": "K", "koniq": "K", "koniq10k": "K", "s": "S", "spaq": "S",
                 "m": "M", "md61": "M", "mt61": "M", "md61base": "M", "mt61base": "M"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_dataset(value) -> str:
    if not isinstance(value, str):
        raise ValueError("dataset must name KonIQ-10k, SPAQ, or MD61/MT61")
    key = value.casefold().replace("-", "").replace("_", "").replace(" ", "")
    if key not in DATASET_NAMES:
        raise ValueError(f"Unsupported dataset {value!r}")
    return DATASET_NAMES[key]


def _identity_digest(values) -> str:
    encoded = json.dumps(sorted(values), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def inspect_training_file(path: Path, stage: str, expected: set[str], *,
                          augmentation: bool, synonym_probability: float,
                          teacher: str, image_hashes: dict | None = None, image_root: Path | None = None) -> tuple[dict, dict]:

    image_hashes = {} if image_hashes is None else image_hashes
    counts, ids, inputs, originals = Counter(), set(), set(), {name: set() for name in expected}
    source_ids = set()

    def image_digest(value, label):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must name a readable image file")
        image = resolve_path(value, path, image_root)
        if not image.is_file():
            raise FileNotFoundError(image)
        if image not in image_hashes:
            image_hashes[image] = file_sha256(image)
        return image_hashes[image]

    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("record must be an object")
                identifier = row.get("id")
                if not isinstance(identifier, str) or not identifier.strip() or identifier in ids:
                    raise ValueError("id must be unique and nonempty")
                ids.add(identifier)
                dataset = canonical_dataset(row.get("dataset"))
                if dataset not in expected:
                    raise ValueError(f"Dataset {dataset} is excluded from this setting")
                if not isinstance(row.get("split"), str) or row["split"].strip().casefold() not in {"train", "training"}:
                    raise ValueError("Every record must explicitly declare split='train' or 'training'")
                for split_field in ("source_split", "official_split"):
                    if split_field in row and (not isinstance(row[split_field], str) or row[split_field].strip().casefold() not in {"train", "training"}):
                        raise ValueError(f"{split_field} contradicts the declared training split")
                edge = row.get("long_edge", 448)
                if type(edge) is not int or edge not in (448, 720):
                    raise ValueError("long_edge must be 448 or 720")
                input_sha = image_digest(row.get("image"), "image")
                original_sha = image_digest(row["original_image"], "original_image") if "original_image" in row else input_sha
                if "original_image_sha256" in row and row["original_image_sha256"] != original_sha:
                    raise ValueError("original_image_sha256 does not match verified original bytes; supply original_image when using prepared inputs")
                inputs.add((dataset, input_sha))
                originals[dataset].add(original_sha)
                if "source_id" in row:
                    if not isinstance(row["source_id"], str) or not row["source_id"].strip():
                        raise ValueError("source_id must be nonempty when supplied")
                    source_ids.add((dataset, row["source_id"]))
                if stage == "caption":
                    if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
                        raise ValueError("Caption records require their explicit description prompt")
                    if not isinstance(row.get("answer"), str) or not row["answer"].strip():
                        raise ValueError("Caption records require the externally supplied answer")
                    declared = row.get("caption_teacher", row.get("teacher_model"))
                    if dataset in {"K", "S"} and declared is not None and str(declared).casefold() != teacher:
                        raise ValueError("Caption teacher metadata contradicts --caption-teacher")
                else:
                    score = row.get("score")
                    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 100:
                        raise ValueError("Score records require a finite numeric score in [0,100]")
                    if augmentation:
                        validate_augmentation_record(row, require_paraphrases=synonym_probability > 0)
                    elif "prompt" in row:
                        if not isinstance(row["prompt"], str) or not row["prompt"].strip():
                            raise ValueError("Score prompt must be nonempty")
                    else:
                        validate_definition(row.get("definition"))
                counts[dataset] += 1
            except (ValueError, TypeError, FileNotFoundError) as error:
                raise ValueError(f"{path}:{number}: {error}") from error
    if set(counts) != expected:
        raise ValueError(f"{stage} datasets must equal {sorted(expected)}; found {sorted(counts)}")
    result = {"path": str(path), "sha256": file_sha256(path), "records": sum(counts.values()),
              "dataset_records": dict(sorted(counts.items())),
              "unique_input_images": len(inputs), "input_images_sha256": _identity_digest(inputs),
              "verified_originals_sha256": _identity_digest((name, sha) for name, hashes in originals.items() for sha in hashes),
              "record_ids_sha256": _identity_digest(ids), "source_ids_sha256": _identity_digest(source_ids),
              "verified_original_images": {name: len(hashes) for name, hashes in sorted(originals.items())}}
    return result, originals


def _validate_base(model_dir: Path) -> dict:
    for marker in ("checkpoint.json", "export.json", "trainable_state.pt", "full_train_state.pt", "adapter_config.json"):
        if (model_dir / marker).exists():
            raise ValueError("--model-dir must be the pretrained backbone, not an OASIS or adapter checkpoint")
    config = model_dir / "config.json"
    content = json.loads(config.read_text(encoding="utf-8"))
    if content.get("model_type") != "qwen3_vl":
        raise ValueError("Expected a Qwen3-VL backbone configuration")
    files = {}
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "preprocessor_config.json", "model.safetensors.index.json"):
        path = model_dir / name
        if path.is_file():
            files[name] = file_sha256(path)
    return {"path": str(model_dir), "configuration_files_sha256": files,
            "declared_role": "pretrained_backbone"}


def _validate_codebook(path: Path) -> str:
    book = json.loads(path.read_text(encoding="utf-8"))
    ids, words = book.get("token_ids"), book.get("codebook")
    if (book.get("score_order") != list(range(0, 101, 10)) or not isinstance(ids, list)
            or len(ids) != 11 or any(type(value) is not int or value < 0 for value in ids)
            or len(set(ids)) != 11 or not isinstance(words, list) or len(words) != 11
            or any(not isinstance(word, str) or not word for word in words)):
        raise ValueError("Expected the fixed eleven-token score codebook")
    return file_sha256(path)


def build_plan(args) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    expected = SETTINGS[args.setting]
    base = _validate_base(args.model_dir)
    codebook_sha = _validate_codebook(args.codebook)
    cache = {}
    caption, caption_images = inspect_training_file(args.caption_train, "caption", expected,
        augmentation=False, synonym_probability=args.synonym_probability, teacher=args.caption_teacher, image_hashes=cache, image_root=args.image_root)
    score, score_images = inspect_training_file(args.score_train, "score", expected,
        augmentation=not args.no_definition_augmentation, synonym_probability=args.synonym_probability,
        teacher=args.caption_teacher, image_hashes=cache, image_root=args.image_root)
    for dataset in expected & {"K", "S"}:
        if caption_images[dataset] != score_images[dataset]:
            raise ValueError(f"Caption and score training must use the same verified {dataset} source images")
    output = args.output_dir
    outputs = {name: str(output / name) for name in ("caption", "caption_merged", "score", "score_merged")}
    common = ["--model-dir", str(args.model_dir), "--batch-size", str(args.batch_size),
              "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
              "--seed", str(args.seed), "--save-steps", str(args.save_steps),
              "--dataloader-workers", str(args.dataloader_workers), "--image-policy", "paper"]
    if args.image_root is not None:
        common += ["--image-root", str(args.image_root)]
    if args.already_prepared:
        common.append("--already-prepared")
    launcher = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={args.gpus}", "-m", "oasis.training"]
    phases = []
    for stage, data in (("caption", args.caption_train), ("score", args.score_train)):
        params = {"learning_rate": getattr(args, f"{stage}_learning_rate"),
                  "weight_decay": getattr(args, f"{stage}_weight_decay"),
                  "rank": getattr(args, f"{stage}_rank"), "alpha": getattr(args, f"{stage}_alpha"),
                  "epochs": getattr(args, f"{stage}_epochs"), "dropout": args.lora_dropout}
        command = launcher + ["--stage", stage, "--train-file", str(data), "--output-dir", outputs[stage]] + common
        for key, flag in (("learning_rate", "--learning-rate"), ("weight_decay", "--weight-decay"),
                          ("rank", "--lora-rank"), ("alpha", "--lora-alpha"), ("epochs", "--epochs"), ("dropout", "--lora-dropout")):
            command += [flag, str(params[key])]
        if stage == "score":
            command += ["--initial-state", str(output / "caption_merged/full_train_state.pt"),
                        "--codebook", str(args.codebook), "--synonym-probability", str(args.synonym_probability),
                        "--name-probability", str(args.name_probability)]
            if args.no_definition_augmentation:
                command.append("--no-definition-augmentation")
        phases.append({"name": stage, "argv": command, "parameters": params})
        phases.append({"name": f"{stage}_export", "argv": [sys.executable, "-m", "oasis.export",
                       "--checkpoint", outputs[stage], "--output", outputs[f"{stage}_merged"]]})
    versions = {"python": sys.version.split()[0]}
    for package in ("torch", "transformers", "peft", "accelerate"):
        try:
            versions[package] = package_metadata.version(package)
        except package_metadata.PackageNotFoundError:
            versions[package] = None
    return {"format": "oasis_iqa_training_plan_v1", "setting": args.setting,
            "base_model": base, "output_dir": str(output), "outputs": outputs,
            "caption_teacher": {"iqa_teacher": args.caption_teacher, "evidence": "user_declaration",
                                "scope": "KonIQ/SPAQ caption records; no descriptions generated by this runner"},
            "inputs": {"caption": caption, "score": score},
            "codebook": {"path": str(args.codebook), "sha256": codebook_sha},
            "execution": {"gpus": args.gpus, "batch_size_per_gpu": args.batch_size,
                          "gradient_accumulation_steps": args.gradient_accumulation_steps,
                          "global_batch_size": args.gpus * args.batch_size * args.gradient_accumulation_steps,
                          "seed": args.seed, "image_policy": "paper", "already_prepared": args.already_prepared,
                          "image_root": str(args.image_root) if args.image_root is not None else None,
                          "augmentation": not args.no_definition_augmentation,
                          "synonym_probability": args.synonym_probability, "name_probability": args.name_probability,
                          "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                          "working_directory": str(Path.cwd().resolve()), "versions": versions},
            "protocol_checks": {"training_splits_only": True, "exact_dataset_sets_both_stages": True,
                                "matching_iqa_source_images": True, "fresh_caption_from_backbone": True,
                                "score_parent": "this_run_caption_export"}, "phases": phases}


def _atomic_json(path: Path, content: dict):
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(content, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _verify_phase_output(plan, phase):
    export = phase.endswith("_export")
    stage = phase.removesuffix("_export")
    folder = Path(plan["outputs"][stage + "_merged" if export else stage])
    weight = folder / ("full_train_state.pt" if export else "trainable_state.pt")
    metadata = folder / ("export.json" if export else "checkpoint.json")
    if not weight.is_file() or weight.stat().st_size == 0:
        raise RuntimeError(f"Phase {phase} did not produce its nonempty weight file")
    content = json.loads(metadata.read_text(encoding="utf-8"))
    if content.get("stage") != stage:
        raise RuntimeError(f"Phase {phase} produced mismatched stage metadata")
    expected_parent = None if stage == "caption" else str(Path(plan["outputs"]["caption_merged"]) / "full_train_state.pt")
    if content.get("initial_state") != expected_parent:
        raise RuntimeError(f"Phase {phase} produced incorrect parent-stage lineage")
    if str(Path(content.get("base_model", "")).resolve()) != plan["base_model"]["path"]:
        raise RuntimeError(f"Phase {phase} used a different backbone")
    return {"weights": str(weight), "weight_bytes": weight.stat().st_size,
            "metadata": str(metadata), "metadata_sha256": file_sha256(metadata)}


def _verify_training_input(plan, stage):
    current, _ = inspect_training_file(Path(plan["inputs"][stage]["path"]), stage,
        SETTINGS[plan["setting"]], augmentation=stage == "score" and plan["execution"]["augmentation"],
        synonym_probability=plan["execution"]["synonym_probability"], teacher=plan["caption_teacher"]["iqa_teacher"], image_root=plan["execution"].get("image_root"))
    if current != plan["inputs"][stage]:
        raise ValueError("External training records or images changed after planning")
    if file_sha256(Path(plan["codebook"]["path"])) != plan["codebook"]["sha256"]:
        raise ValueError("The score codebook changed after planning")


def execute_plan(plan: dict):

    root = Path(plan["output_dir"])
    root.mkdir(parents=True, exist_ok=False)
    logs = root / "logs"
    logs.mkdir()
    _atomic_json(root / "experiment.json", plan)
    status = {"status": "RUNNING", "completed_phases": [], "artifacts": {}}
    try:
        for stage in ("caption", "score"):
            _verify_training_input(plan, stage)
        for phase in plan["phases"]:
            status.update(phase=phase["name"], updated_at=datetime.now(timezone.utc).isoformat())
            _atomic_json(root / "status.json", status)
            if _validate_base(Path(plan["base_model"]["path"])) != plan["base_model"]:
                raise ValueError("Backbone configuration changed after planning")
            if phase["name"] in {"caption", "score"}:
                _verify_training_input(plan, phase["name"])
            with (logs / f"{phase['name']}.log").open("x", encoding="utf-8") as handle:
                subprocess.run(phase["argv"], check=True, stdout=handle, stderr=subprocess.STDOUT,
                               cwd=plan["execution"]["working_directory"])
            status["artifacts"][phase["name"]] = _verify_phase_output(plan, phase["name"])
            status["completed_phases"].append(phase["name"])
        status.update(status="COMPLETE", phase=None, final_state=str(root / "score_merged/full_train_state.pt"))
    except BaseException as error:
        status.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        status["updated_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(root / "status.json", status)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=tuple(SETTINGS), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--caption-train", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--score-train", type=Path, required=True)
    parser.add_argument("--caption-teacher", choices=("gpt-5.4",), required=True)
    parser.add_argument("--codebook", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--dataloader-workers", type=int, default=0)
    parser.add_argument("--already-prepared", action="store_true")
    parser.add_argument("--no-definition-augmentation", action="store_true")
    parser.add_argument("--synonym-probability", type=float, default=.5)
    parser.add_argument("--name-probability", type=float, default=.5)
    parser.add_argument("--lora-dropout", type=float, default=.05)
    for stage, learning_rate, decay, rank, alpha in (("caption", 4e-5, .05, 16, 32), ("score", 2e-5, 0., 128, 256)):
        parser.add_argument(f"--{stage}-epochs", type=float, default=1.)
        parser.add_argument(f"--{stage}-learning-rate", type=float, default=learning_rate)
        parser.add_argument(f"--{stage}-weight-decay", type=float, default=decay)
        parser.add_argument(f"--{stage}-rank", type=int, default=rank)
        parser.add_argument(f"--{stage}-alpha", type=int, default=alpha)
    args = parser.parse_args(argv)
    for name in ("model_dir", "caption_train", "score_train", "codebook", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.image_root is not None:
        args.image_root = args.image_root.expanduser().resolve()
    positive = ("gpus", "batch_size", "gradient_accumulation_steps", "save_steps", "caption_epochs", "score_epochs",
                "caption_learning_rate", "score_learning_rate", "caption_rank", "score_rank", "caption_alpha", "score_alpha")
    if any(not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0 for key in positive):
        parser.error("GPU/batch/step counts, epochs, learning rates and LoRA ranks/alphas must be positive")
    if args.dataloader_workers < 0 or not math.isfinite(args.lora_dropout) or not 0 <= args.lora_dropout < 1:
        parser.error("Invalid worker count or LoRA dropout")
    if any(not math.isfinite(getattr(args, key)) or getattr(args, key) < 0 for key in ("caption_weight_decay", "score_weight_decay")):
        parser.error("Weight decay must be finite and nonnegative")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in (args.synonym_probability, args.name_probability)):
        parser.error("Augmentation probabilities must be finite and in [0,1]")
    return args


def main(argv=None):
    args = parse_args(argv)
    plan = build_plan(args)
    display = dict(plan)
    display["phases"] = [dict(phase, command=shlex.join(phase["argv"])) for phase in plan["phases"]]
    print(json.dumps(display, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    if args.execute:
        execute_plan(plan)


if __name__ == "__main__":
    main()
