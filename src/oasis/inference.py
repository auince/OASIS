from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from .paths import resolve_path


def read_requests(path: Path, *, image_only: bool = False, image_root=None) -> list[dict]:

    rows, seen = [], set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error.msg}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number}: expected an object")
            identifier = row.get("id")
            if not isinstance(identifier, str) or not identifier.strip() or identifier in seen:
                raise ValueError(f"Line {line_number}: id must be a unique nonempty string")
            if not isinstance(row.get("image"), str) or not row["image"].strip():
                raise ValueError(f"Line {line_number}: image must name a readable local file")
            row["image"] = str(resolve_path(row["image"], path, root=image_root))
            if not Path(row["image"]).is_file():
                raise ValueError(f"Line {line_number}: image must name a readable local file")
            has_prompt = "prompt" in row
            has_definition = "definition" in row
            if image_only:
                if has_definition or (has_prompt and row["prompt"] != ""):
                    raise ValueError(
                        f"Line {line_number}: --image-only requires no definition and "
                        "an omitted prompt or exactly an empty prompt string")
            else:
                if has_prompt == has_definition:
                    raise ValueError(f"Line {line_number}: supply exactly one of prompt or definition")
                if has_prompt and (not isinstance(row["prompt"], str) or not row["prompt"].strip()):
                    raise ValueError(f"Line {line_number}: prompt must be a nonempty string")
                if has_definition and not isinstance(row["definition"], dict):
                    raise ValueError(f"Line {line_number}: definition must be an object")
            edge = row.get("long_edge", 448)
            if type(edge) is not int or edge not in (448, 720):
                raise ValueError(f"Line {line_number}: long_edge must be 448 or 720")
            seen.add(identifier)
            rows.append(row)
    if not rows:
        raise ValueError("The input contains no requests")
    return rows


def read_codebook(path: Path) -> dict:
    book = json.loads(path.read_text(encoding="utf-8"))
    anchors = book.get("score_order", book.get("anchors"))
    ids = book.get("token_ids")
    if anchors != list(range(0, 101, 10)):
        raise ValueError("The codebook must contain anchors 0, 10, ..., 100 in order")
    if (not isinstance(ids, list) or len(ids) != 11
            or any(type(value) is not int or value < 0 for value in ids)
            or len(set(ids)) != 11):
        raise ValueError("The codebook must contain eleven distinct nonnegative token IDs")
    return {"token_ids": ids, "score_order": anchors}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codebook", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--image-only", action="store_true")
    parser.add_argument("--image-policy", choices=("paper", "archive-jpeg95"), default="paper")
    parser.add_argument("--already-prepared", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    rows = read_requests(args.input, image_only=args.image_only, image_root=args.image_root)
    book = read_codebook(args.codebook)
    state_file = args.state_file or args.model_dir / "full_train_state.pt"
    if not state_file.is_file():
        raise FileNotFoundError(state_file)

    import torch
    from .images import prepare_image
    from .modeling import load_backbone, load_processor
    from .prompts import build_prompt


    for row in rows:
        if "definition" in row:
            row["prompt"] = build_prompt(row["definition"])
    processor = load_processor(args.model_dir)
    vocab_size = len(processor.tokenizer)
    if max(book["token_ids"]) >= vocab_size:
        raise ValueError("A score token ID exceeds the tokenizer vocabulary")
    model = load_backbone(args.model_dir, device=args.device, initial_state=state_file,
                          dtype=args.dtype, attn_implementation="sdpa")
    model.eval()
    ids = torch.tensor(book["token_ids"], dtype=torch.long, device=args.device)
    axis = torch.tensor(book["score_order"], dtype=torch.float32, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent,
                                         prefix=f".{args.output.name}.", suffix=".tmp",
                                         delete=False) as handle, torch.inference_mode():
            temporary_path = Path(handle.name)
            for row in rows:
                picture = prepare_image(row["image"], long_edge=row.get("long_edge", 448),
                                        already_prepared=args.already_prepared,
                                        policy=args.image_policy)
                try:
                    content = [{"type": "image", "image": picture}]
                    if not args.image_only:
                        content.append({"type": "text", "text": row["prompt"]})
                    chat = [{"role": "user", "content": content}]
                    inputs = processor.apply_chat_template(
                        chat, tokenize=True, add_generation_prompt=True,
                        return_dict=True, return_tensors="pt").to(args.device)
                    logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1]
                    probabilities = logits.float().index_select(-1, ids).softmax(-1)
                    if not torch.isfinite(probabilities).all().item():
                        raise RuntimeError(f"Nonfinite score probabilities for {row['id']}")
                    result = {"id": row["id"], "score": float((probabilities @ axis).item()),
                              "probabilities": probabilities[0].cpu().tolist()}
                    handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                finally:
                    picture.close()
            handle.flush()
            os.fsync(handle.fileno())

        os.link(temporary_path, args.output)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
