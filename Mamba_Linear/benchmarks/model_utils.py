"""Unified model loader for Mamba2 (baseline) and Mamba2MC (cache variant).

The two models share the same backbone weights; only `Mamba2MC` adds the
extra parameters `W` and `online_bias` for the memory-cache mixing path.
A single fine-tuned checkpoint can therefore be loaded into either class.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch

from mamba2 import Mamba2Config, Mamba2LMHeadModel
from mamba2_mc import Mamba2MCLMHeadModel


@dataclass
class ModelSpec:
    name: str                       # display name, e.g. "Mamba2MC-seg64-cache16"
    family: str                     # "mamba2" or "mamba2mc"
    segment_size: int = 64
    max_cached_segments: int = 16
    min_history_segments: int = 1


def build_config(d_model: int = 2048, n_layer: int = 48, vocab_size: int = 50288) -> Mamba2Config:
    return Mamba2Config(d_model=d_model, n_layer=n_layer, vocab_size=vocab_size)


def load_model(spec: ModelSpec, ckpt_path: str, device: str = "cuda") -> torch.nn.Module:
    """Build the model and load weights. ckpt_path is the dir containing pytorch_model.bin."""
    cfg = build_config()
    if spec.family == "mamba2":
        model = Mamba2LMHeadModel(cfg)
    elif spec.family == "mamba2mc":
        model = Mamba2MCLMHeadModel(
            cfg,
            segment_size=spec.segment_size,
            max_cached_segments=spec.max_cached_segments,
            min_history_segments=spec.min_history_segments,
        )
    else:
        raise ValueError(f"Unknown model family: {spec.family}")

    sd_path = os.path.join(ckpt_path, "pytorch_model.bin")
    state_dict = torch.load(sd_path, map_location="cpu")
    result = model.load_state_dict(state_dict, strict=False)
    print(
        f"[load_model] {spec.name}: missing={len(result.missing_keys)}, "
        f"unexpected={len(result.unexpected_keys)}"
    )
    model = model.to(device)
    model.eval()
    return model


def model_forward_logits(model, input_ids):
    """Wrap forward call so both Mamba2 and Mamba2MC return (logits, _)."""
    out = model(input_ids)
    if isinstance(out, tuple):
        return out[0]
    return out
