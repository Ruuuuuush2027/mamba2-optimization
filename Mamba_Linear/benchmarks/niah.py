"""Needle-in-a-Haystack benchmark.

Build long contexts of `target_len` tokens by repeating a filler corpus,
insert a "needle" sentence (e.g. "The magic number is 749231.") at
relative depth `d` in {0, 0.1, ..., 1.0}, then ask the model to recall
the magic number. Score = exact match on the digits.

Returns a 2D grid of recall scores indexed by (ctx_len, depth) suitable
for a heatmap.
"""
from __future__ import annotations

import random
import re

import torch
from datasets import load_dataset
from tqdm import tqdm


FILLER_SENT = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again. "
)

NEEDLE_TEMPLATE = "The magic number is {key}. Remember this number: {key}."
QUESTION = "\n\nQuestion: What is the magic number? Answer with just the digits.\nAnswer:"


def _build_haystack(tokenizer, target_len: int) -> list[int]:
    """Build a token sequence of approximately `target_len` tokens."""
    text = ""
    while True:
        text += FILLER_SENT
        ids = tokenizer(text, return_tensors="pt").input_ids[0]
        if len(ids) >= target_len:
            return ids[:target_len].tolist()


def _insert_needle(haystack_ids: list[int], needle_ids: list[int], depth: float) -> list[int]:
    """Insert needle at fractional depth `depth` ∈ [0, 1]."""
    pos = int(len(haystack_ids) * depth)
    return haystack_ids[:pos] + needle_ids + haystack_ids[pos:]


@torch.no_grad()
def _greedy_short(model, tokenizer, prompt_ids: torch.Tensor, max_new: int = 16) -> str:
    """Greedy decoding (full-forward) — short generation only."""
    device = prompt_ids.device
    out_ids: list[int] = []
    cur = prompt_ids
    for _ in range(max_new):
        out = model(cur)
        logits = out[0] if isinstance(out, tuple) else out
        nxt = int(logits[0, -1].argmax().item())
        out_ids.append(nxt)
        cur = torch.cat([cur, torch.tensor([[nxt]], device=device)], dim=1)
    return tokenizer.decode(out_ids, skip_special_tokens=True)


@torch.no_grad()
def eval_niah(model, tokenizer, device: str = "cuda",
              ctx_lens: list[int] | None = None,
              depths: list[float] | None = None,
              n_keys_per_cell: int = 3,
              seed: int = 0) -> dict:
    ctx_lens = ctx_lens or [512, 1024, 2048, 4096]
    depths = depths or [0.0, 0.25, 0.5, 0.75, 1.0]
    rng = random.Random(seed)

    grid = {}  # grid[ctx_len][depth] = recall in [0, 1]
    for ctx_len in ctx_lens:
        # Build the haystack once per ctx_len (deterministic)
        haystack = _build_haystack(tokenizer, ctx_len)
        question_ids = tokenizer(QUESTION, return_tensors="pt").input_ids[0].tolist()

        per_depth = {}
        for depth in depths:
            hits = 0
            for _ in range(n_keys_per_cell):
                key = f"{rng.randint(100000, 999999)}"
                needle = tokenizer(NEEDLE_TEMPLATE.format(key=key),
                                    return_tensors="pt").input_ids[0].tolist()
                full = _insert_needle(haystack, needle, depth) + question_ids
                ids = torch.tensor([full], device=device)
                pred = _greedy_short(model, tokenizer, ids, max_new=12)
                m = re.search(r"\d{4,7}", pred)
                if m and m.group() == key:
                    hits += 1
            per_depth[depth] = hits / n_keys_per_cell
        grid[ctx_len] = per_depth

    # Flatten to JSON-friendly form
    return {
        "ctx_lens": ctx_lens,
        "depths": depths,
        "grid": {str(k): {str(d): v for d, v in row.items()} for k, row in grid.items()},
        "avg": sum(v for row in grid.values() for v in row.values())
              / sum(len(r) for r in grid.values()),
    }
