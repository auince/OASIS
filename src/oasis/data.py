from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .images import prepare_image
from .paths import resolve_path
from .prompts import build_prompt
from .augmentation import DefinitionAugmenter, validate_augmentation_record, validate_definition


class ImageScoreDataset(Dataset):

    def __init__(self, path: str | Path, stage: str, definition_augmentation: bool = False,
                 synonym_probability: float = 0.5, name_probability: float = 0.5,
                 image_root: str | Path | None = None):
        if stage not in {"caption", "score"}:
            raise ValueError("Stage must be caption or score")
        if definition_augmentation and stage != "score":
            raise ValueError("Definition augmentation applies only to score training")
        self.definition_augmentation = definition_augmentation
        augmenter = DefinitionAugmenter(synonym_probability, name_probability)
        self._augmenter = augmenter if definition_augmentation else None
        self.input = Path(path).expanduser().resolve()
        self.rows = []
        seen = set()
        with self.input.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Line {line_number}: training record must be an object")
                if not all(isinstance(row.get(key), str) and row[key].strip()
                           for key in ("id", "image")):
                    raise ValueError(f"Line {line_number}: nonempty id and image required")
                row["image"] = str(resolve_path(row["image"], self.input, root=image_root))
                for image_field in ("source_image", "original_image"):
                    if row.get(image_field):
                        row[image_field] = str(resolve_path(row[image_field], self.input, root=image_root))
                if definition_augmentation:
                    try:
                        validate_augmentation_record(row, require_paraphrases=synonym_probability > 0)
                    except ValueError as error:
                        raise ValueError(f"Line {line_number}: {error}") from error
                    row["prompt"] = build_prompt(row["definition"])
                elif stage == "score" and "prompt" not in row and "definition" in row:
                    validate_definition(row["definition"])
                    row["prompt"] = build_prompt(row["definition"])
                if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
                    requirement = "nonempty caption prompt required" if stage == "caption" else "nonempty prompt or structured definition required"
                    raise ValueError(f"Line {line_number}: {requirement}")
                if row["id"] in seen:
                    raise ValueError(f"Line {line_number}: duplicate id {row['id']!r}")
                seen.add(row["id"])
                edge = row.get("long_edge", 448)
                if type(edge) is not int or edge not in (448, 720):
                    raise ValueError(f"Line {line_number}: long_edge must be 448 or 720")
                if stage == "caption":
                    if not isinstance(row.get("answer"), str) or not row["answer"].strip():
                        raise ValueError(f"Line {line_number}: caption answer required")
                else:
                    value = row.get("score")
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise ValueError(f"Line {line_number}: numeric score required")
                    if not math.isfinite(value) or not 0 <= value <= 100:
                        raise ValueError(f"Line {line_number}: score must be finite and in [0, 100]")
                self.rows.append(row)
        if not self.rows:
            raise ValueError("Training file has no records")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return self._augmenter(row) if self._augmenter is not None else deepcopy(row)


class MultimodalCollator:

    def __init__(self, processor, stage: str, codebook: dict | None = None,
                 already_prepared: bool = False, max_response_tokens: int = 2048,
                 image_policy: str = "paper"):
        self.processor = processor
        self.stage = stage
        self.codebook = codebook
        self.already_prepared = already_prepared
        if image_policy not in {"paper", "archive-jpeg95"}:
            raise ValueError("image_policy must be paper or archive-jpeg95")
        self.image_policy = image_policy
        self.max_response_tokens = max_response_tokens
        if stage == "score" and codebook is None:
            raise ValueError("Score training requires a codebook")
        if stage not in {"caption", "score"}:
            raise ValueError("Stage must be caption or score")

    def __call__(self, rows):
        tokenizer = self.processor.tokenizer
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define an EOS token")
        packed = []
        for row in rows:
            image = prepare_image(row["image"], long_edge=row.get("long_edge", 448),
                                  already_prepared=self.already_prepared, policy=self.image_policy)
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": row["prompt"]},
            ]}]
            try:
                prefix = self.processor.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt")
            finally:
                image.close()
            if self.stage == "caption":
                answer = tokenizer.encode(row["answer"], add_special_tokens=False)
                answer.append(tokenizer.eos_token_id)
            else:
                index = min(10, math.floor(float(row["score"]) / 10 + 0.5))
                answer = [self.codebook["token_ids"][index]]
            if len(answer) > self.max_response_tokens:
                raise ValueError(f"Response for {row['id']!r} exceeds max_response_tokens")
            response = torch.tensor(answer, dtype=torch.long)
            prefix_ids = prefix["input_ids"][0]
            if prefix_ids.numel() == 0:
                raise ValueError("Chat template produced an empty prefix")
            ids = torch.cat((prefix_ids, response))
            labels = torch.cat((torch.full_like(prefix_ids, -100), response))
            packed.append((ids, labels, prefix))
        longest = max(ids.numel() for ids, _, _ in packed)
        padding_id = tokenizer.pad_token_id
        if padding_id is None:
            padding_id = tokenizer.eos_token_id
        input_ids = torch.full((len(rows), longest), padding_id, dtype=torch.long)
        labels = torch.full_like(input_ids, -100)
        attention_mask = torch.zeros_like(input_ids)
        for index, (ids, targets, _) in enumerate(packed):
            length = ids.numel()
            input_ids[index, :length] = ids
            labels[index, :length] = targets
            attention_mask[index, :length] = 1
        result = {
            "input_ids": input_ids, "labels": labels, "attention_mask": attention_mask,
            "pixel_values": torch.cat([prefix["pixel_values"] for _, _, prefix in packed]),
            "image_grid_thw": torch.cat([prefix["image_grid_thw"] for _, _, prefix in packed]),
        }
        if self.stage == "score":
            result["gold_scores"] = torch.tensor([row["score"] for row in rows], dtype=torch.float32)
        return result
