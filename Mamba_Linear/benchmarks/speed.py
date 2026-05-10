"""Inference latency / throughput / VRAM measurement."""
from __future__ import annotations

import gc
import time

import torch


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


@torch.no_grad()
def measure_prefill(model, seq_len: int, device: str = "cuda",
                    n_warmup: int = 1, n_runs: int = 3) -> dict:
    """Time a single batch=1 prefill of `seq_len` tokens."""
    vocab = model.args.vocab_size if hasattr(model, "args") else 50288
    dummy = torch.randint(0, vocab, (1, seq_len), device=device)

    for _ in range(n_warmup):
        _ = model(dummy)
    _sync()
    _reset_mem()

    times = []
    for _ in range(n_runs):
        _sync()
        t0 = time.perf_counter()
        _ = model(dummy)
        _sync()
        times.append(time.perf_counter() - t0)

    peak_mb = (torch.cuda.max_memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else 0.0
    avg = sum(times) / len(times)
    return {
        "seq_len": seq_len,
        "prefill_time_s": avg,
        "prefill_tokens_per_s": seq_len / avg,
        "peak_vram_mb": peak_mb,
    }


@torch.no_grad()
def measure_decode(model, prompt_len: int, n_new: int = 64, device: str = "cuda") -> dict:
    """Time `n_new` autoregressive steps after a `prompt_len` prefill.

    Uses `step()` (Mamba2MC) or the (b,1) forward path for Mamba2 baseline.
    Both should be O(1) per token.
    """
    vocab = model.args.vocab_size if hasattr(model, "args") else 50288
    dummy = torch.randint(0, vocab, (1, prompt_len), device=device)

    # Prefill to populate cache.
    out = model(dummy)
    cache = out[1] if isinstance(out, tuple) else None

    _sync()
    _reset_mem()
    t0 = time.perf_counter()

    last_tok = dummy[:, -1:]
    for _ in range(n_new):
        if hasattr(model, "step") and cache is not None:
            _, cache = model.step(last_tok, cache)
        else:
            # Fallback for plain Mamba2: pass cache=h_list
            out = model(last_tok, cache)
            cache = out[1] if isinstance(out, tuple) else None
    _sync()
    elapsed = time.perf_counter() - t0
    peak_mb = (torch.cuda.max_memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else 0.0
    return {
        "prompt_len": prompt_len,
        "n_new": n_new,
        "decode_time_s": elapsed,
        "decode_tokens_per_s": n_new / elapsed,
        "peak_vram_mb": peak_mb,
    }


def sweep_speed(model, seq_lens: list[int], n_new: int = 64, device: str = "cuda") -> dict:
    rows = []
    for sl in seq_lens:
        try:
            pre = measure_prefill(model, sl, device=device)
            dec = measure_decode(model, sl, n_new=n_new, device=device)
            rows.append({**pre, **{k: v for k, v in dec.items() if k not in pre}})
        except torch.cuda.OutOfMemoryError:
            print(f"[speed] OOM at seq_len={sl}, stopping sweep")
            rows.append({"seq_len": sl, "oom": True})
            break
    return {"rows": rows}
