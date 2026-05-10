"""LongBench evaluation (subset).

Uses THUDM/LongBench from HuggingFace. We default to a small set of
English subtasks that cover QA, summarization, and few-shot,
plus the official length-aware truncation.

For each example we (1) truncate the input to the model's context limit
keeping the head + tail of the prompt, (2) do greedy decoding for
`max_new_tokens` (per-task, defined by LongBench), then (3) score with
the official metric.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Callable

import torch
from datasets import load_dataset
from tqdm import tqdm


# Per-task generation lengths from LongBench official config
TASK_MAX_NEW_TOKENS = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "triviaqa": 32,
    "samsum": 128,
    "passage_count": 32,
    "passage_retrieval_en": 32,
}

# Default subset (you can override via CLI). Picked for breadth + small generation budgets.
DEFAULT_TASKS = [
    "qasper",            # QA over scientific papers
    "multifieldqa_en",   # mixed-domain QA
    "hotpotqa",          # multi-hop QA
    "gov_report",        # summarization
    "passage_retrieval_en",  # retrieval
]


# ---------- metrics ----------
def _normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(pred: str, ground_truth: str) -> float:
    pred_tokens = _normalize_answer(pred).split()
    gt_tokens = _normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def best_subspan_em(pred: str, ground_truth: str) -> float:
    return float(_normalize_answer(ground_truth) in _normalize_answer(pred))


# Map each task to its scoring function (simplified — official LongBench
# uses Rouge for summarization; we use F1 as a fast proxy)
TASK_METRIC: dict[str, Callable[[str, str], float]] = {
    "qasper": f1_score,
    "multifieldqa_en": f1_score,
    "hotpotqa": f1_score,
    "narrativeqa": f1_score,
    "gov_report": f1_score,
    "qmsum": f1_score,
    "multi_news": f1_score,
    "passage_retrieval_en": best_subspan_em,
    "triviaqa": f1_score,
    "samsum": f1_score,
    "2wikimqa": f1_score,
    "passage_count": best_subspan_em,
}


# ---------- generation ----------
@torch.no_grad()
def _greedy_generate(model, tokenizer, prompt_ids: torch.Tensor,
                     max_new_tokens: int, eos_token_id: int) -> str:
    """Greedy decoding via repeated forward passes (works for both Mamba2 and Mamba2MC).

    Note: this is the simplest, most portable path — it re-runs the full
    forward each step rather than using the incremental cache, which is
    fine for benchmarking quality but slow. For pure speed numbers see
    benchmarks/speed.py which uses generate()/step().
    """
    device = prompt_ids.device
    generated: list[int] = []
    cur = prompt_ids.clone()
    for _ in range(max_new_tokens):
        out = model(cur)
        logits = out[0] if isinstance(out, tuple) else out
        next_token = int(logits[0, -1].argmax().item())
        if next_token == eos_token_id:
            break
        generated.append(next_token)
        cur = torch.cat([cur, torch.tensor([[next_token]], device=device)], dim=1)
    return tokenizer.decode(generated, skip_special_tokens=True)


def _truncate_prompt(input_ids: torch.Tensor, max_len: int) -> torch.Tensor:
    """LongBench-style middle truncation: keep head + tail."""
    if input_ids.shape[1] <= max_len:
        return input_ids
    half = max_len // 2
    return torch.cat([input_ids[:, :half], input_ids[:, -half:]], dim=1)


@torch.no_grad()
def eval_longbench(model, tokenizer, device: str = "cuda",
                   tasks: list[str] | None = None,
                   max_examples_per_task: int = 50,
                   max_input_len: int = 4096) -> dict:
    """Run LongBench subset and return per-task + average score."""
    tasks = tasks or DEFAULT_TASKS
    results = {}

    for task in tasks:
        try:
            ds = load_dataset("THUDM/LongBench", task, split="test", trust_remote_code=True)
        except Exception as e:
            print(f"[longbench] skipping {task}: {e}")
            results[task] = None
            continue

        ds = ds.select(range(min(max_examples_per_task, len(ds))))
        max_new = TASK_MAX_NEW_TOKENS.get(task, 64)
        metric = TASK_METRIC.get(task, f1_score)

        scores = []
        for ex in tqdm(ds, desc=f"longbench:{task}"):
            prompt = ex.get("input") or ex.get("prompt") or ""
            context = ex.get("context", "")
            full_text = f"{context}\n\n{prompt}".strip()
            ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)
            ids = _truncate_prompt(ids, max_input_len)

            pred = _greedy_generate(model, tokenizer, ids, max_new, tokenizer.eos_token_id or 0)
            answers = ex.get("answers", [])
            best = max((metric(pred, a) for a in answers), default=0.0)
            scores.append(best)

        results[task] = sum(scores) / len(scores) if scores else None

    valid_scores = [v for v in results.values() if v is not None]
    results["_avg"] = sum(valid_scores) / len(valid_scores) if valid_scores else None
    return results
