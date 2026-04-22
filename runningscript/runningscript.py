#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_full_context_prompt(sample):
    turns = sample["turns"]
    query = sample["query_turn"]["content"]

    context = "\n".join(t["content"] for t in turns)
    prompt = (
        f"{context}\n\n"
        f"Question: {query}\n"
        f"Answer briefly:"
    )
    return prompt


def build_incremental_prompt(sample):
    """
    Text-level incremental prompt.
    It appends turns one by one in order, then asks the final question.
    If your checkpoint supports cache-based incremental decoding, the script
    will try to use it. Otherwise it falls back to the concatenated prompt.
    """
    turns = sample["turns"]
    query = sample["query_turn"]["content"]

    context_parts = []
    for t in turns:
        context_parts.append(t["content"])
    context = "\n".join(context_parts)

    prompt = (
        f"{context}\n\n"
        f"Question: {query}\n"
        f"Answer briefly:"
    )
    return prompt


@torch.no_grad()
def generate_answer(model, tokenizer, prompt, max_new_tokens=32):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id,
    )

    gen_ids = output[0][input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text


@torch.no_grad()
def generate_incremental_answer(model, tokenizer, sample, max_new_tokens=32):
    """
    Best-effort incremental mode:
    1) Prefill turn by turn.
    2) Try to reuse cache if the model exposes it.
    3) Fall back to one-shot prompt if cache path fails.
    """
    device = model.device
    turns = [t["content"] for t in sample["turns"]]
    query = sample["query_turn"]["content"]

    past = None
    cache_ok = True

    for turn in turns:
        turn_text = turn + "\n"
        turn_ids = tokenizer(turn_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        if turn_ids.numel() == 0:
            continue

        try:
            out = model(
                input_ids=turn_ids,
                use_cache=True,
                past_key_values=past,
            )
            past = getattr(out, "past_key_values", None)
            if past is None:
                past = getattr(out, "cache_params", None)
            if past is None:
                cache_ok = False
                break
        except Exception:
            cache_ok = False
            break

    if cache_ok and past is not None:
        query_text = f"\nQuestion: {query}\nAnswer briefly:"
        query_ids = tokenizer(query_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

        # Try a few possible cache argument names
        for cache_kw in ("past_key_values", "cache_params"):
            try:
                gen = model.generate(
                    input_ids=query_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tokenizer.eos_token_id,
                    **{cache_kw: past},
                )
                gen_ids = gen[0][query_ids.shape[-1]:]
                return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            except Exception:
                pass

    # Fallback: same final prompt, but no cache path
    prompt = build_incremental_prompt(sample)
    return generate_answer(model, tokenizer, prompt, max_new_tokens=max_new_tokens)


def run_mode(model, tokenizer, data, mode, out_path, max_new_tokens=32, limit=None):
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for sample in data:
            sample_mode = sample.get("eval_mode", "unknown")
            if sample_mode != mode:
                continue

            if limit is not None and n >= limit:
                break

            if mode == "full_context":
                prompt = build_full_context_prompt(sample)
                pred = generate_answer(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
            elif mode == "incremental":
                pred = generate_incremental_answer(model, tokenizer, sample, max_new_tokens=max_new_tokens)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            f.write(json.dumps({
                "id": sample["id"],
                "prediction": pred
            }, ensure_ascii=False) + "\n")

            n += 1
            print(f"[{mode}] {n}: {sample['id']} -> {pred}")

    print(f"Saved {n} predictions to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="mango_v2.jsonl")
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--mode", type=str, choices=["incremental", "full_context", "both"], default="both")
    ap.add_argument("--out_dir", type=str, default="preds_out")
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    data = load_jsonl(args.data)

    if args.mode == "both":
        run_mode(
            model, tokenizer, data,
            mode="incremental",
            out_path=str(out_dir / "preds_incremental.jsonl"),
            max_new_tokens=args.max_new_tokens,
            limit=args.limit,
        )
        run_mode(
            model, tokenizer, data,
            mode="full_context",
            out_path=str(out_dir / "preds_full_context.jsonl"),
            max_new_tokens=args.max_new_tokens,
            limit=args.limit,
        )
    else:
        run_mode(
            model, tokenizer, data,
            mode=args.mode,
            out_path=str(out_dir / f"preds_{args.mode}.jsonl"),
            max_new_tokens=args.max_new_tokens,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()