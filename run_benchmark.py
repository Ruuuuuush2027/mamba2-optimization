import argparse
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

    parser.add_argument("--model-type", type=str, default="Mamba2MC", choices=["Mamba2", "Mamba2MC"])
    parser.add_argument("--model-id", type=str, default="state-spaces/mamba2-1.3b")
    parser.add_argument("--tokenizer-id", type=str, default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--cache-dir", type=str, default="./huggingface_cache")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Checkpoint dir containing pytorch_model.bin")

    # Keep these aligned with finetune.py model-loading knobs.
    parser.add_argument("--mc-segment-size", type=int, default=64)
    parser.add_argument("--mc-max-cached-segments", type=int, default=16)
    parser.add_argument("--mc-backprop-history", action="store_true")

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
    parser.add_argument("--skip-piqa", action="store_true", help="Skip PIQA evaluation.")
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
    else:
        model = load_our_model(
            args.model_id,
            device=device,
            cache_dir=args.cache_dir,
            segment_size=args.mc_segment_size,
            max_cached_segments=args.mc_max_cached_segments,
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


def score_text_logprob(model, tokenizer, device, text: str) -> float:
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        logits, _ = model(input_ids)
    shift_logits = logits[:, :-1]
    shift_labels = input_ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.sum().item()


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
        s1 = score_text_logprob(model, tokenizer, device, context + " " + example["sol1"])
        s2 = score_text_logprob(model, tokenizer, device, context + " " + example["sol2"])
        pred = 0 if s1 > s2 else 1
        if pred == example["label"]:
            correct += 1
        progress.update(1)
        progress.set_postfix(acc=f"{(correct / i):.4f}")
    progress.close()

    acc = correct / total if total > 0 else 0.0
    print(f"PIQA Accuracy: {acc:.6f}")
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


if __name__ == "__main__":
    main()
