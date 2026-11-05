from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile


def read_metadata(checkpoint: Path) -> dict:
    metadata = json.loads((checkpoint / "checkpoint.json").read_text(encoding="utf-8"))
    if metadata.get("format") != "oasis_trainable_v1":
        raise ValueError("Expected an OASIS oasis_trainable_v1 checkpoint")
    for name in ("base_model", "rank", "alpha", "dropout", "stage", "initial_state", "keys"):
        if name not in metadata:
            raise ValueError(f"Checkpoint metadata is missing {name}")
    if type(metadata["rank"]) is not int or metadata["rank"] < 1:
        raise ValueError("Checkpoint rank must be a positive integer")
    if metadata["stage"] not in ("stage1", "stage2", "caption", "score"):
        raise ValueError("Checkpoint stage must be caption/score or stage1/stage2")
    if not isinstance(metadata["keys"], list) or len(metadata["keys"]) != len(set(metadata["keys"])):
        raise ValueError("Checkpoint keys must list each trained parameter exactly once")
    return metadata


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--initial-state", type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    metadata = read_metadata(args.checkpoint)
    state_path = args.checkpoint / "trainable_state.pt"
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    base_model = args.base_model or Path(metadata["base_model"])
    initial_state = args.initial_state or metadata["initial_state"]
    if metadata["stage"] in ("stage2", "score") and not initial_state:
        raise ValueError("A stage2 checkpoint requires its merged stage1 initial_state")
    if initial_state and not Path(initial_state).is_file():
        raise FileNotFoundError(f"Parent stage state not found: {initial_state}")

    import torch
    from .modeling import attach_lora, load_backbone, load_processor

    model = load_backbone(base_model, device=None, initial_state=initial_state,
                          dtype="bfloat16", attn_implementation="sdpa")
    model = attach_lora(model, rank=metadata["rank"], alpha=metadata["alpha"],
                        dropout=metadata["dropout"])
    state = torch.load(state_path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict):
        raise ValueError("trainable_state.pt must contain a parameter dictionary")
    expected = {name for name, value in model.named_parameters() if value.requires_grad}
    if set(state) != expected or set(metadata["keys"]) != expected:
        missing, extra = sorted(expected - set(state)), sorted(set(state) - expected)
        raise ValueError(f"Incomplete or mismatched trainable state: missing={missing[:8]}, "
                         f"extra={extra[:8]}; checkpoint metadata must match the exact trainable key set")
    restored = model.load_state_dict(state, strict=False)
    if restored.unexpected_keys:
        raise ValueError(f"Unexpected checkpoint parameters: {restored.unexpected_keys[:8]}")
    del state
    model = model.merge_and_unload()
    model.eval()
    processor_source = args.checkpoint if (args.checkpoint / "preprocessor_config.json").is_file() else base_model
    processor = load_processor(processor_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:

        torch.save(model.state_dict(), staging / "full_train_state.pt")
        model.config.save_pretrained(staging)
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.save_pretrained(staging)
        processor.save_pretrained(staging)
        manifest = {"format": "oasis_merged_v1", "source_checkpoint": str(args.checkpoint.resolve()),
                    "stage": metadata["stage"], "base_model": str(base_model),
                    "initial_state": str(initial_state) if initial_state else None,
                    "weight_file": "full_train_state.pt"}
        (staging / "export.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite {args.output}")
        staging.rename(args.output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()
