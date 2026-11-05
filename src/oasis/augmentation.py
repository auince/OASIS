from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from numbers import Real
import os
import random

from .prompts import build_prompt

FIELDS = ("name", "description", "positive_evidence", "negative_evidence", "score_guidance")
NEUTRAL_NAME = "the specified criterion"


def validate_definition(definition, label="definition"):
    if not isinstance(definition, dict) or set(definition) != set(FIELDS):
        raise ValueError(f"{label} must contain exactly {', '.join(FIELDS)}")
    for field in FIELDS:
        if not isinstance(definition[field], str) or not definition[field].strip():
            raise ValueError(f"{label}.{field} must be a nonempty string")
    if "\n" in definition["name"] or "\r" in definition["name"]:
        raise ValueError(f"{label}.name must be a single line")


def validate_augmentation_record(row, require_paraphrases=True):

    original = row.get("definition")
    validate_definition(original)
    paraphrases = row.get("paraphrases", [])
    if not isinstance(paraphrases, list) or (require_paraphrases and not paraphrases):
        raise ValueError("Positive synonym probability requires at least one reviewed paraphrase")
    seen = {tuple(original[key] for key in FIELDS)}
    for index, variant in enumerate(paraphrases):
        label = f"paraphrases[{index}]"
        validate_definition(variant, label)
        if variant["name"] != original["name"]:
            raise ValueError(f"{label}.name must match the original criterion name")
        if variant["score_guidance"] != original["score_guidance"]:
            raise ValueError(f"{label}.score_guidance must be identical to the original")
        key = tuple(variant[field] for field in FIELDS)
        if key in seen:
            raise ValueError(f"{label} duplicates the original or another paraphrase")
        seen.add(key)
    aliases = row.get("aliases", [NEUTRAL_NAME])
    if not isinstance(aliases, list) or not aliases:
        raise ValueError("aliases must be a nonempty list when provided")
    if any(not isinstance(alias, str) or not alias.strip() or "\n" in alias or "\r" in alias
           for alias in aliases):
        raise ValueError("Each alias must be a nonempty single-line string")
    if len(aliases) != len(set(aliases)) or original["name"] in aliases:
        raise ValueError("Aliases must be distinct and differ from the canonical name")
    if "prompt" in row and row["prompt"] != build_prompt(original):
        raise ValueError("When augmentation is enabled, prompt must exactly match build_prompt(definition)")


class DefinitionAugmenter:

    def __init__(self, synonym_probability=0.5, name_probability=0.5):
        for label, value in (("synonym_probability", synonym_probability),
                             ("name_probability", name_probability)):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{label} must be finite and within [0, 1]")
        self.synonym_probability = float(synonym_probability)
        self.name_probability = float(name_probability)
        self._context = None
        self._rng = None

    def _random_stream(self):
        import torch
        from torch.utils.data import get_worker_info

        worker = get_worker_info()
        worker_id = worker.id if worker is not None else -1
        seed = worker.seed if worker is not None else torch.initial_seed()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = int(os.environ.get("RANK", "0"))
        context = (os.getpid(), seed, rank, worker_id)
        if context != self._context:
            material = f"oasis-definition-augmentation|{seed}|{rank}|{worker_id}".encode()
            mixed_seed = int.from_bytes(hashlib.sha256(material).digest()[:16], "big")
            self._rng = random.Random(mixed_seed)
            self._context = context
        return self._rng

    def __call__(self, row):
        rng = self._random_stream()
        rewrite = rng.random() < self.synonym_probability
        replace_name = rng.random() < self.name_probability
        result = deepcopy(row)
        selected = rng.choice(row["paraphrases"]) if rewrite else row["definition"]
        definition = dict(selected)
        if replace_name:
            definition["name"] = rng.choice(row.get("aliases", [NEUTRAL_NAME]))
        result["prompt"] = build_prompt(definition)
        return result
