"""WikiText-2 perplexity (sliding window)."""
from __future__ import annotations

import torch
from datasets import load_dataset
from tqdm import tqdm

from .model_utils import model_forward_logits


@torch.no_grad()
def eval_wikitext(model, tokenizer, device: str = "cuda",
                  max_length: int = 1024, stride: int = 512,
                  max_chunks: int | None = None) -> dict:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"][0]

    nlls, n_tokens = [], 0
    iterator = range(0, len(input_ids) - max_length, stride)
    if max_chunks is not None:
        iterator = list(iterator)[:max_chunks]

    for i in tqdm(iterator, desc="wikitext"):
        chunk = input_ids[i:i + max_length].unsqueeze(0).to(device)
        logits = model_forward_logits(model, chunk)
        shift_logits = logits[:, :-1]
        shift_labels = chunk[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="mean",
        )
        nlls.append(loss.float())
        n_tokens += shift_labels.numel()

    ppl = torch.exp(torch.stack(nlls).mean()).item()
    return {"ppl": ppl, "n_tokens": n_tokens, "n_chunks": len(nlls)}
