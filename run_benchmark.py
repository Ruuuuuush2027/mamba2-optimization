import argparse
import random
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark fine-tuned Mamba2 / Mamba2MC checkpoints.")

    parser.add_argument("--model-type", type=str, default="Mamba2MC", choices=["Mamba2", "Mamba2MC", "Mamba2MCSelect"])
    parser.add_argument("--model-id", type=str, default="state-spaces/mamba2-1.3b")
    parser.add_argument("--tokenizer-id", type=str, default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--cache-dir", type=str, default="./huggingface_cache")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Checkpoint dir containing pytorch_model.bin")

    # Keep these aligned with finetune.py model-loading knobs.
    parser.add_argument("--mc-segment-size", type=int, default=64)
    parser.add_argument("--mc-max-cached-segments", type=int, default=16)
    parser.add_argument("--mc-backprop-history", action="store_true")
    parser.add_argument("--mc-select-keep-top-k", type=int, default=8)
    parser.add_argument("--mc-select-score-threshold", type=float, default=-1.0)

    # WikiText perplexity settings.
    parser.add_argument("--wikitext-config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--wikitext-split", type=str, default="test")
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=1024)

    # PIQA settings.
    parser.add_argument("--piqa-dataset-id", type=str, default="baber/piqa")
    parser.add_argument(
        "--piqa-local-path",
        type=str,
        default="",
        help="Local PIQA data path (saved HF dataset dir or parquet file/dir).",
    )
    parser.add_argument("--piqa-split", type=str, default="validation")
    parser.add_argument("--max-piqa-samples", type=int, default=0, help="0 means evaluate full split")
    parser.add_argument(
        "--piqa-length-normalize",
        action="store_true",
        help="Normalize PIQA continuation log-prob by continuation token count.",
    )
    parser.add_argument("--skip-piqa", action="store_true", help="Skip PIQA evaluation.")

    # Long-context Needle-In-A-Haystack (NIAH) settings.
    parser.add_argument("--skip-niah", action="store_true", help="Skip long-context NIAH benchmark.")
    parser.add_argument("--niah-num-examples", type=int, default=20, help="Number of NIAH trials.")
    parser.add_argument(
        "--niah-context-tokens",
        type=int,
        default=8192,
        help="Approximate context length in tokens for each NIAH prompt.",
    )
    parser.add_argument(
        "--niah-needle-position",
        type=str,
        default="middle",
        choices=["early", "middle", "late", "random"],
        help="Where to place the needle sentence in the long context.",
    )
    parser.add_argument("--niah-seed", type=int, default=42)
    parser.add_argument("--niah-max-new-tokens", type=int, default=16)
    return parser.parse_args()


def load_mamba2_model(model_id: str, device, cache_dir: str):
    from mamba2 import Mamba2LMHeadModel

    return Mamba2LMHeadModel.from_pretrained(model_id, device=device, cache_dir=cache_dir)


def load_our_model(
    model_id: str,
    device,
    cache_dir: str,
    segment_size: int,
    max_cached_segments: int,
    detach_cached_segments: bool,
):
    from mamba2_mc import Mamba2MCLMHeadModel

    return Mamba2MCLMHeadModel.from_pretrained(
        model_id,
        device=device,
        cache_dir=cache_dir,
        segment_size=segment_size,
        max_cached_segments=max_cached_segments,
        detach_cached_segments=detach_cached_segments,
    )


def load_our_select_model(
    model_id: str,
    device,
    cache_dir: str,
    segment_size: int,
    max_cached_segments: int,
    keep_top_k: int,
    score_threshold: float,
    detach_cached_segments: bool,
):
    from mc_select import Mamba2MCSelectLMHeadModel

    return Mamba2MCSelectLMHeadModel.from_pretrained(
        model_id,
        device=device,
        cache_dir=cache_dir,
        segment_size=segment_size,
        max_cached_segments=max_cached_segments,
        keep_top_k=keep_top_k,
        score_threshold=score_threshold,
        detach_cached_segments=detach_cached_segments,
    )


def load_model_and_tokenizer(args, device):
    ckpt_dir = Path(args.checkpoint_path)
    weights_path = ckpt_dir / "pytorch_model.bin"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"--checkpoint-path does not exist: {ckpt_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint weights not found: {weights_path}")

    print("Loading model...")
    if args.model_type == "Mamba2":
        model = load_mamba2_model(args.model_id, device=device, cache_dir=args.cache_dir)
    elif args.model_type == "Mamba2MC":
        model = load_our_model(
            args.model_id,
            device=device,
            cache_dir=args.cache_dir,
            segment_size=args.mc_segment_size,
            max_cached_segments=args.mc_max_cached_segments,
            detach_cached_segments=(not args.mc_backprop_history),
        )
    else:
        model = load_our_select_model(
            args.model_id,
            device=device,
            cache_dir=args.cache_dir,
            segment_size=args.mc_segment_size,
            max_cached_segments=args.mc_max_cached_segments,
            keep_top_k=args.mc_select_keep_top_k,
            score_threshold=args.mc_select_score_threshold,
            detach_cached_segments=(not args.mc_backprop_history),
        )

    print(f"Loading checkpoint weights from {weights_path}")
    state_dict = torch.load(weights_path, map_location=device)
    # Keep strict=True to fail fast on architecture/config mismatch.
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    tokenizer_source = args.tokenizer_id
    if (ckpt_dir / "tokenizer.json").exists() or (ckpt_dir / "tokenizer_config.json").exists():
        tokenizer_source = str(ckpt_dir)
        print(f"Loading tokenizer from checkpoint dir: {ckpt_dir}")
    else:
        print(f"Loading tokenizer from --tokenizer-id: {tokenizer_source}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def run_wikitext_perplexity(model, tokenizer, device, args) -> float:
    print(f"Running WikiText ({args.wikitext_config}, split={args.wikitext_split})...")
    dataset = load_dataset("wikitext", args.wikitext_config, split=args.wikitext_split, cache_dir=args.cache_dir)
    text = "\n\n".join(dataset["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"][0]

    if args.max_length <= 1:
        raise ValueError("--max-length must be > 1")
    if args.stride <= 0:
        raise ValueError("--stride must be > 0")
    if len(input_ids) <= args.max_length:
        raise ValueError("Input sequence is shorter than --max-length; lower --max-length.")

    nlls = []
    for i in tqdm(range(0, len(input_ids) - args.max_length, args.stride), desc="WikiText"):
        chunk = input_ids[i : i + args.max_length].unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = model(chunk)
        shift_logits = logits[:, :-1]
        shift_labels = chunk[:, 1:]
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="mean",
        )
        nlls.append(loss)

    ppl = torch.exp(torch.stack(nlls).mean()).item()
    print(f"WikiText PPL: {ppl:.6f}")
    return ppl


def score_text_logprob(model, tokenizer, device, prefix: str, continuation: str, length_normalize: bool) -> float:
    full_text = prefix + " " + continuation
    inputs = tokenizer(full_text, return_tensors="pt").to(device)
    prefix_ids = tokenizer(prefix, return_tensors="pt")["input_ids"]
    prefix_len = int(prefix_ids.size(1))

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")

    # Mamba2 implementation expects seq_len to be divisible by chunk_size.
    chunk_size = getattr(getattr(model, "args", None), "chunk_size", 64)
    seq_len = input_ids.size(1)
    remainder = seq_len % chunk_size
    if remainder != 0:
        pad_len = chunk_size - remainder
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        input_ids = F.pad(input_ids, (0, pad_len), value=pad_token_id)
        if attention_mask is None:
            attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
        attention_mask = F.pad(attention_mask, (0, pad_len), value=0)

    with torch.no_grad():
        logits, _ = model(input_ids)
    shift_logits = logits[:, :-1]
    shift_labels = input_ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    valid_mask = torch.ones_like(token_log_probs, dtype=torch.bool)
    # Keep only continuation tokens (conditioned on the prefix).
    # shift_labels index j corresponds to original token position i=j+1.
    start_idx = max(prefix_len - 1, 0)
    if start_idx > 0:
        valid_mask[:, :start_idx] = False
    if attention_mask is not None:
        valid_mask = valid_mask & attention_mask[:, 1:].bool()

    selected = token_log_probs[valid_mask]
    if selected.numel() == 0:
        return float("-inf")
    total = selected.sum().item()
    if length_normalize:
        return total / float(selected.numel())
    return total


def load_piqa_dataset(args):
    if args.piqa_local_path:
        local_path = Path(args.piqa_local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"--piqa-local-path does not exist: {local_path}")

        # Try loading as a local HF dataset saved via save_to_disk().
        try:
            local_ds = load_from_disk(str(local_path))
            if hasattr(local_ds, "keys"):
                if args.piqa_split in local_ds:
                    return local_ds[args.piqa_split]
                first_split = next(iter(local_ds.keys()))
                print(
                    f"PIQA local dataset does not have split '{args.piqa_split}', "
                    f"using '{first_split}' instead."
                )
                return local_ds[first_split]
            return local_ds
        except Exception:
            pass

        # Fall back to local parquet loading.
        if local_path.is_file():
            data_files = str(local_path)
        else:
            parquet_files = sorted(str(p) for p in local_path.rglob("*.parquet"))
            if not parquet_files:
                raise ValueError(
                    f"No parquet files found under --piqa-local-path: {local_path}"
                )
            data_files = parquet_files
        return load_dataset("parquet", data_files=data_files, split="train", cache_dir=args.cache_dir)

    return load_dataset(args.piqa_dataset_id, split=args.piqa_split, cache_dir=args.cache_dir)


def run_piqa_accuracy(model, tokenizer, device, args) -> float:
    print(f"Running PIQA (split={args.piqa_split})...")
    try:
        dataset = load_piqa_dataset(args)
    except Exception as exc:
        msg = str(exc)
        if "Dataset scripts are no longer supported" in msg:
            print(
                "Skipping PIQA: this datasets version blocks script-based PIQA loading. "
                "Set --piqa-local-path to local parquet/save_to_disk data, or use --piqa-dataset-id "
                "with a parquet-backed PIQA mirror."
            )
            return float("nan")
        raise
    if args.max_piqa_samples > 0:
        dataset = dataset.select(range(min(args.max_piqa_samples, len(dataset))))

    correct = 0
    total = len(dataset)
    progress = tqdm(total=total, desc="PIQA", unit="ex")
    for i, example in enumerate(dataset, start=1):
        context = example["goal"]
        s1 = score_text_logprob(
            model, tokenizer, device, context, example["sol1"], length_normalize=args.piqa_length_normalize
        )
        s2 = score_text_logprob(
            model, tokenizer, device, context, example["sol2"], length_normalize=args.piqa_length_normalize
        )
        pred = 0 if s1 > s2 else 1
        if pred == example["label"]:
            correct += 1
        progress.update(1)
        progress.set_postfix(acc=f"{(correct / i):.4f}")
    progress.close()

    acc = correct / total if total > 0 else 0.0
    print(f"PIQA Accuracy: {acc:.6f}")
    return acc


def _make_needle_code(rng: random.Random) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(rng.choice(alphabet) for _ in range(8))


def _normalize_answer(text: str) -> str:
    text = text.strip().upper()
    m = re.search(r"[A-Z0-9]{8}", text)
    return m.group(0) if m else text


def _build_niah_prompt(tokenizer, target_tokens: int, needle_code: str, position: str, rng: random.Random) -> str:
    filler = (
        "This is background context for a retrieval test. "
        "Most lines are irrelevant and should be ignored. "
    )
    filler_tokens = tokenizer(filler, add_special_tokens=False)["input_ids"]
    filler_repeats = max(8, target_tokens // max(1, len(filler_tokens)))
    chunks = [filler for _ in range(filler_repeats)]

    needle_sentence = (
        f"Important memo: the access code is {needle_code}. "
        "Remember this exact code.\n"
    )

    if position == "early":
        idx = max(1, len(chunks) // 10)
    elif position == "middle":
        idx = len(chunks) // 2
    elif position == "late":
        idx = max(0, (len(chunks) * 9) // 10)
    else:
        idx = rng.randint(0, len(chunks))

    chunks.insert(idx, needle_sentence)
    body = "".join(chunks)

    question = (
        "\nQuestion: What is the exact access code from the memo above? "
        "Answer with the code only.\nAnswer:"
    )
    return body + question


def run_niah_benchmark(model, tokenizer, device, args) -> float:
    print(
        f"Running Long NIAH (examples={args.niah_num_examples}, context_tokens~{args.niah_context_tokens}, "
        f"position={args.niah_needle_position})..."
    )
    rng = random.Random(args.niah_seed)
    correct = 0
    progress = tqdm(total=args.niah_num_examples, desc="NIAH", unit="ex")

    for i in range(args.niah_num_examples):
        needle = _make_needle_code(rng)
        prompt = _build_niah_prompt(
            tokenizer=tokenizer,
            target_tokens=args.niah_context_tokens,
            needle_code=needle,
            position=args.niah_needle_position,
            rng=rng,
        )
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)[0]

        generated_piece = []
        for token_id, _ in model.generate(
            input_ids,
            max_new_length=args.niah_max_new_tokens,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0,
        ):
            generated_piece.append(tokenizer.decode([token_id]))

        pred = _normalize_answer("".join(generated_piece))
        gold = _normalize_answer(needle)
        if pred == gold:
            correct += 1

        progress.update(1)
        progress.set_postfix(acc=f"{(correct / (i + 1)):.4f}")

    progress.close()
    acc = correct / max(1, args.niah_num_examples)
    print(f"NIAH Accuracy: {acc:.6f}")
    return acc


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    model, tokenizer = load_model_and_tokenizer(args, device)
    run_wikitext_perplexity(model, tokenizer, device, args)
    if args.skip_piqa:
        print("Skipping PIQA (--skip-piqa set).")
    else:
        run_piqa_accuracy(model, tokenizer, device, args)
    if args.skip_niah:
        print("Skipping NIAH (--skip-niah set.")
    else:
        run_niah_benchmark(model, tokenizer, device, args)


if __name__ == "__main__":
    main()
