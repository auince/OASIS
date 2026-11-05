import json
import os
from pathlib import Path

import torch
from accelerate import init_empty_weights
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration


def load_processor(model_dir):
    return AutoProcessor.from_pretrained(
        str(model_dir), size={"shortest_edge": 1, "longest_edge": 5_000_000}
    )


def load_backbone(model_dir, device=None, initial_state=None, dtype="bfloat16",
                  attn_implementation="sdpa"):
    dtypes = {"bfloat16": torch.bfloat16, "float32": torch.float32}
    if dtype not in dtypes:
        raise ValueError("dtype must be bfloat16 or float32")
    torch_dtype = dtypes[dtype]
    if initial_state is not None:
        path = Path(initial_state)
        if not path.is_file():
            raise FileNotFoundError(path)
        config = AutoConfig.from_pretrained(str(model_dir))
        if config.model_type != "qwen3_vl":
            raise ValueError("This implementation requires a qwen3_vl configuration")
        config._attn_implementation = attn_implementation
        with init_empty_weights():
            model = Qwen3VLForConditionalGeneration(config)
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        if not isinstance(state, dict) or not all(torch.is_tensor(v) for v in state.values()):
            raise ValueError("Expected a merged, flat full-model state dictionary")
        model.load_state_dict(state, strict=True, assign=True)
        del state
        model.tie_weights()
        if any(p.is_meta for p in model.parameters()):
            raise RuntimeError("Full state did not materialize every parameter")
        model = model.to(dtype=torch_dtype)
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_dir), dtype=torch_dtype, attn_implementation=attn_implementation
        )
    if device is not None:
        model = model.to(device)
    return model


def attach_lora(model, rank, alpha, dropout):
    if rank < 1 or alpha < 1 or not 0 <= dropout < 1:
        raise ValueError("Invalid LoRA rank, alpha or dropout")
    model = get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                        "up_proj", "down_proj"],
    ))
    visual_parameters = 0
    for name, parameter in model.named_parameters():
        if ".visual." in name:
            parameter.requires_grad_(True)
            visual_parameters += parameter.numel()
    if not visual_parameters:
        raise RuntimeError("No visual parameters found; refusing incomplete training")
    return model


def save_trainable_checkpoint(model, processor, output_dir, metadata):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {name: parameter.detach().cpu() for name, parameter in model.named_parameters()
             if parameter.requires_grad}
    if not state:
        raise ValueError("Model has no trainable parameters")
    temporary = output_dir / "trainable_state.pt.tmp"
    torch.save(state, temporary)
    os.replace(temporary, output_dir / "trainable_state.pt")
    info = dict(metadata)
    info.update(format="oasis_trainable_v1", keys=sorted(state))
    temporary_json = output_dir / "checkpoint.json.tmp"
    temporary_json.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_json, output_dir / "checkpoint.json")
    processor.save_pretrained(output_dir)
    model.config.save_pretrained(output_dir)
