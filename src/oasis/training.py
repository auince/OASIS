from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments, set_seed

from .data import ImageScoreDataset, MultimodalCollator
from .losses import score_loss
from .modeling import (attach_lora, load_backbone, load_processor,
                       save_trainable_checkpoint)


def load_score_codebook(path: str | Path, tokenizer) -> dict:

    book = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = book.get("token_ids")
    words = book.get("codebook")
    if book.get("score_order") != list(range(0, 101, 10)):
        raise ValueError("Score anchors must be 0, 10, ..., 100")
    if not isinstance(ids, list) or len(ids) != 11 or len(set(ids)) != 11:
        raise ValueError("Codebook must contain eleven distinct token IDs")
    if not isinstance(words, list) or len(words) != 11:
        raise ValueError("Codebook must contain eleven token strings")
    for word, token_id in zip(words, ids):
        if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
            raise ValueError("Codebook token IDs must be nonnegative integers")
        if not isinstance(word, str) or tokenizer.encode(word, add_special_tokens=False) != [token_id]:
            raise ValueError(f"Codebook string {word!r} does not encode to token {token_id}")
    if tokenizer.eos_token_id in ids:
        raise ValueError("A score token cannot be the EOS token")
    return book


class TwoStageTrainer(Trainer):

    def __init__(self, *args, stage: str, processor, checkpoint_metadata: dict,
                 codebook: dict | None = None, **kwargs):
        self.stage = stage
        self.processor_for_save = processor
        self.checkpoint_metadata = checkpoint_metadata
        self.codebook = codebook
        super().__init__(*args, **kwargs)

        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        scores = inputs.pop("gold_scores", None)
        positions = torch.nonzero((labels[:, 1:] != -100).any(0), as_tuple=False).flatten() + 1
        if positions.numel() == 0:
            raise ValueError("Batch contains no assistant supervision")


        outputs = model(**inputs, use_cache=False, logits_to_keep=positions - 1)
        targets = labels[:, positions]
        valid = targets != -100
        if self.stage == "caption":
            loss = F.cross_entropy(outputs.logits[valid].float(), targets[valid])
        else:
            if not (valid.sum(-1) == 1).all() or scores is None:
                raise ValueError("Score batches must supervise exactly one token per example")
            ids = torch.tensor(self.codebook["token_ids"], device=targets.device)
            if not (targets[valid].unsqueeze(-1) == ids).any(-1).all():
                raise ValueError("Supervised score token is outside the fixed codebook")
            anchors = torch.tensor(self.codebook["score_order"], device=targets.device,
                                   dtype=torch.float32)
            loss, metrics = score_loss(outputs.logits[valid], scores, ids, anchors)
            self.log({key: value.item() for key, value in metrics.items()})
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss")
        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir=None, state_dict=None):


        output = Path(output_dir or self.args.output_dir)
        save_trainable_checkpoint(self.model, self.processor_for_save, output,
                                  self.checkpoint_metadata)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["caption", "score"], required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path)
    parser.add_argument("--codebook", type=Path)
    parser.add_argument("--already-prepared", action="store_true")
    parser.add_argument("--image-policy", choices=("paper", "archive-jpeg95"), default="paper")
    parser.add_argument("--no-definition-augmentation", action="store_true")
    parser.add_argument("--synonym-probability", type=float, default=0.5)
    parser.add_argument("--name-probability", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--dataloader-workers", type=int, default=0)
    parser.add_argument("--max-response-tokens", type=int, default=2048)
    args = parser.parse_args(argv)
    if args.stage == "score" and (args.initial_state is None or args.codebook is None):
        parser.error("Score training requires --initial-state and --codebook")
    caption = args.stage == "caption"
    defaults = {"learning_rate": 4e-5 if caption else 2e-5,
                "weight_decay": 0.05 if caption else 0.0,
                "lora_rank": 16 if caption else 128}
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.lora_alpha is None:
        args.lora_alpha = 2 * args.lora_rank
    positive = ("learning_rate", "lora_rank", "lora_alpha", "batch_size",
                "gradient_accumulation_steps", "epochs", "save_steps", "max_response_tokens")
    if any(getattr(args, key) <= 0 for key in positive):
        parser.error("Learning rate, ranks, batch sizes, epochs and step/token limits must be positive")
    if (not all(math.isfinite(value) for value in (args.learning_rate, args.weight_decay, args.epochs, args.lora_dropout))
            or args.weight_decay < 0 or not 0 <= args.lora_dropout < 1 or args.dataloader_workers < 0):
        parser.error("Invalid weight decay, LoRA dropout, or worker count")
    if any(not math.isfinite(value) or not 0 <= value <= 1
           for value in (args.synonym_probability, args.name_probability)):
        parser.error("Augmentation probabilities must be finite and in [0, 1]")
    return args


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    args = parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError("Use a new output directory; existing training output is preserved")
    set_seed(args.seed)
    processor = load_processor(args.model_dir)
    book = load_score_codebook(args.codebook, processor.tokenizer) if args.stage == "score" else None
    augment = args.stage == "score" and not args.no_definition_augmentation
    dataset = ImageScoreDataset(args.train_file, args.stage,
                                definition_augmentation=augment,
                                synonym_probability=args.synonym_probability,
                                name_probability=args.name_probability,
                                image_root=args.image_root)
    model = load_backbone(args.model_dir, initial_state=args.initial_state,
                          dtype="bfloat16", attn_implementation="sdpa")
    model = attach_lora(model, rank=args.lora_rank, alpha=args.lora_alpha,
                        dropout=args.lora_dropout)
    model.config.use_cache = False
    metadata = {
        "format": "oasis_trainable_v1", "stage": args.stage,
        "base_model": str(args.model_dir.resolve()),
        "initial_state": str(args.initial_state.resolve()) if args.initial_state else None,
        "rank": args.lora_rank, "alpha": args.lora_alpha, "dropout": args.lora_dropout,
        "train_file": str(args.train_file.resolve()),
        "train_file_sha256": file_sha256(args.train_file),
        "records": len(dataset), "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay, "epochs": args.epochs, "seed": args.seed,
        "batch_size_per_device": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "already_prepared": args.already_prepared,
        "image_policy": args.image_policy,
        "definition_augmentation": {
            "enabled": augment,
            "sampling": "fresh_on_each_dataset_access",
            "synonym_probability": args.synonym_probability if augment else 0.0,
            "name_probability": args.name_probability if augment else 0.0,
        },
        "score_codebook": book,
    }
    training_args = TrainingArguments(
        output_dir=str(args.output_dir), num_train_epochs=args.epochs,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        lr_scheduler_type="cosine", warmup_ratio=0.03,
        seed=args.seed, data_seed=args.seed, logging_steps=1,
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=None,
        report_to=[], remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_workers,
        dataloader_drop_last=False, ddp_find_unused_parameters=False,
    )
    metadata["world_size"] = training_args.world_size
    metadata["effective_batch_size"] = (args.batch_size * args.gradient_accumulation_steps
                                        * training_args.world_size)
    trainer = TwoStageTrainer(
        model=model, args=training_args, train_dataset=dataset,
        data_collator=MultimodalCollator(processor, args.stage, book,
                                        already_prepared=args.already_prepared,
                                        image_policy=args.image_policy,
                                        max_response_tokens=args.max_response_tokens),
        stage=args.stage, processor=processor, codebook=book,
        checkpoint_metadata=metadata,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    trainer.save_state()


if __name__ == "__main__":
    main()
