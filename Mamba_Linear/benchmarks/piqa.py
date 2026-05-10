"""PIQA 0-shot accuracy via length-normalized log-likelihood scoring."""
from __future__ import annotations

import torch
from datasets import load_dataset
from tqdm import tqdm

from .model_utils import model_forward_logits


@torch.no_grad()
def _score(model, tokenizer, text: str, device: str) -> float:
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    logits = model_forward_logits(model, input_ids)
    shift_logits = logits[:, :-1]
    shift_labels = input_ids[:, 1:]
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    # length-normalized to avoid bias toward shorter completions
    return token_log_probs.sum().item() / max(token_log_probs.numel(), 1)


@torch.no_grad()
def eval_piqa(model, tokenizer, device: str = "cuda",
              max_examples: int | None = None) -> dict:
    dataset = load_dataset("piqa", split="validation", trust_remote_code=True)
    if max_examples is not None:
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    correct = 0
    for example in tqdm(dataset, desc="piqa"):
        s1 = _score(model, tokenizer, example["goal"] + " " + example["sol1"], device)
        s2 = _score(model, tokenizer, example["goal"] + " " + example["sol2"], device)
        pred = 0 if s1 > s2 else 1
        if pred == example["label"]:
            correct += 1

    return {"accuracy": correct / len(dataset), "n_examples": len(dataset)}
