"""Unified benchmark runner for Mamba2 vs Mamba2MC.

Examples
--------
# Quick smoke test (small subsets):
python run_benchmark.py --tasks wikitext piqa --models baseline mc_default --max-examples 50

# Full main comparison (default params):
python run_benchmark.py --tasks wikitext piqa longbench niah speed \
        --models baseline mc_default

# Sweep segment_size:
python run_benchmark.py --tasks wikitext --sweep-segment-size 32 64 128 256

# Sweep cache slots:
python run_benchmark.py --tasks wikitext niah --sweep-cache-slots 4 8 16 32

All results are written as JSON to ./results/<run_name>.json so the
plot_and_report.py script can pick them up later.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from benchmarks.model_utils import ModelSpec, load_model
from benchmarks.wikitext import eval_wikitext
from benchmarks.piqa import eval_piqa
from benchmarks.longbench import eval_longbench, DEFAULT_TASKS
from benchmarks.niah import eval_niah
from benchmarks.speed import sweep_speed


DEFAULT_CKPT = "mamba2-ckpts/checkpoints/mamba2-finetune/Mamba2MC-final"
RESULTS_DIR = Path("results")


# ---------- model presets ----------
def make_specs(model_keys: list[str], seg: int = 64, cache: int = 16) -> list[ModelSpec]:
    presets = {
        "baseline":   ModelSpec(name="Mamba2-baseline", family="mamba2"),
        "mc_default": ModelSpec(name=f"Mamba2MC-seg{seg}-c{cache}", family="mamba2mc",
                                segment_size=seg, max_cached_segments=cache),
    }
    return [presets[k] for k in model_keys]


def run_one(spec: ModelSpec, ckpt: str, tokenizer, args, device: str) -> dict:
    print(f"\n{'='*60}\n  {spec.name}\n{'='*60}")
    model = load_model(spec, ckpt, device=device)
    out: dict = {
        "model": spec.__dict__,
        "tasks": {},
        "timestamp": time.time(),
    }

    if "wikitext" in args.tasks:
        print("\n[wikitext]")
        out["tasks"]["wikitext"] = eval_wikitext(
            model, tokenizer, device=device,
            max_length=args.wt_max_length, stride=args.wt_stride,
            max_chunks=args.max_chunks,
        )

    if "piqa" in args.tasks:
        print("\n[piqa]")
        out["tasks"]["piqa"] = eval_piqa(
            model, tokenizer, device=device, max_examples=args.max_examples,
        )

    if "longbench" in args.tasks:
        print("\n[longbench]")
        out["tasks"]["longbench"] = eval_longbench(
            model, tokenizer, device=device,
            tasks=args.longbench_tasks or DEFAULT_TASKS,
            max_examples_per_task=args.lb_max_per_task,
            max_input_len=args.lb_max_input_len,
        )

    if "niah" in args.tasks:
        print("\n[niah]")
        out["tasks"]["niah"] = eval_niah(
            model, tokenizer, device=device,
            ctx_lens=args.niah_ctx_lens,
            depths=args.niah_depths,
            n_keys_per_cell=args.niah_keys,
        )

    if "speed" in args.tasks:
        print("\n[speed]")
        out["tasks"]["speed"] = sweep_speed(
            model, seq_lens=args.speed_seq_lens, n_new=args.speed_n_new, device=device,
        )

    # Free GPU memory before next model
    del model
    torch.cuda.empty_cache()
    return out


def save_json(data: dict, name: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(data, indent=2, default=str))
    print(f"  -> saved {path}")
    return path


# ---------- argparse ----------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--models", nargs="+", default=["baseline", "mc_default"],
                   choices=["baseline", "mc_default"],
                   help="Which model preset(s) to run in main mode")
    p.add_argument("--tasks", nargs="+",
                   default=["wikitext", "piqa"],
                   choices=["wikitext", "piqa", "longbench", "niah", "speed"])

    # WikiText
    p.add_argument("--wt-max-length", type=int, default=1024)
    p.add_argument("--wt-stride", type=int, default=512)
    p.add_argument("--max-chunks", type=int, default=None,
                   help="cap WikiText chunks for fast iteration")

    # PIQA
    p.add_argument("--max-examples", type=int, default=None)

    # LongBench
    p.add_argument("--longbench-tasks", nargs="+", default=None)
    p.add_argument("--lb-max-per-task", type=int, default=50)
    p.add_argument("--lb-max-input-len", type=int, default=4096)

    # NIAH
    p.add_argument("--niah-ctx-lens", nargs="+", type=int,
                   default=[512, 1024, 2048, 4096])
    p.add_argument("--niah-depths", nargs="+", type=float,
                   default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--niah-keys", type=int, default=3)

    # Speed
    p.add_argument("--speed-seq-lens", nargs="+", type=int,
                   default=[512, 1024, 2048, 4096, 8192])
    p.add_argument("--speed-n-new", type=int, default=64)

    # MC hyperparam sweeps (mutually exclusive with --models for clarity)
    p.add_argument("--sweep-segment-size", nargs="+", type=int, default=None)
    p.add_argument("--sweep-cache-slots", nargs="+", type=int, default=None)
    return p


def main():
    args = build_parser().parse_args()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # ---- Sweep mode (segment_size or cache_slots) ----
    if args.sweep_segment_size:
        for seg in args.sweep_segment_size:
            spec = ModelSpec(name=f"Mamba2MC-seg{seg}-c16",
                             family="mamba2mc",
                             segment_size=seg, max_cached_segments=16)
            res = run_one(spec, args.ckpt, tokenizer, args, args.device)
            save_json(res, f"sweep_seg_{seg}")
        return

    if args.sweep_cache_slots:
        for c in args.sweep_cache_slots:
            spec = ModelSpec(name=f"Mamba2MC-seg64-c{c}",
                             family="mamba2mc",
                             segment_size=64, max_cached_segments=c)
            res = run_one(spec, args.ckpt, tokenizer, args, args.device)
            save_json(res, f"sweep_cache_{c}")
        return

    # ---- Main comparison mode ----
    for spec in make_specs(args.models):
        res = run_one(spec, args.ckpt, tokenizer, args, args.device)
        save_json(res, f"main_{spec.family}_{spec.name}".replace("/", "_"))


if __name__ == "__main__":
    main()
