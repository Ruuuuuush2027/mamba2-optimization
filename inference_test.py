import argparse
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer

from mamba2 import Mamba2LMHeadModel
from mamba2_mc import Mamba2MCLMHeadModel
try:
    from mamba2_mc_select import Mamba2MCSelectLMHeadModel
except ModuleNotFoundError:
    from mc_select import Mamba2MCSelectLMHeadModel


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference test for Mamba2/Mamba2MC checkpoints.")
    parser.add_argument("--model-type", type=str, default="Mamba2", choices=["Mamba2", "Mamba2MC", "Mamba2MCSelect"])
    parser.add_argument("--model-id", type=str, default="state-spaces/mamba2-1.3b")
    parser.add_argument("--tokenizer-id", type=str, default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--cache-dir", type=str, default="./huggingface_cache")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints/mamba2-finetune/Mamba2-final",
        help="Directory containing pytorch_model.bin from fine-tuning.",
    )
    parser.add_argument("--prompt", type=str, default="Mamba is")
    parser.add_argument("--max-new-length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    # Mamba2MC-specific runtime args
    parser.add_argument("--mc-segment-size", type=int, default=64)
    parser.add_argument("--mc-max-cached-segments", type=int, default=16)
    parser.add_argument("--mc-backprop-history", action="store_true")
    parser.add_argument("--mc-select-keep-top-k", type=int, default=8)
    parser.add_argument("--mc-select-score-threshold", type=float, default=-1.0)
    return parser.parse_args()


def load_model(args: argparse.Namespace, device: torch.device):
    if args.model_type == "Mamba2":
        model = Mamba2LMHeadModel.from_pretrained(
            args.model_id, device=device, cache_dir=args.cache_dir
        )
    elif args.model_type == "Mamba2MC":
        model = Mamba2MCLMHeadModel.from_pretrained(
            args.model_id,
            device=device,
            cache_dir=args.cache_dir,
            segment_size=args.mc_segment_size,
            max_cached_segments=args.mc_max_cached_segments,
            detach_cached_segments=(not args.mc_backprop_history),
        )
    else:
        model = Mamba2MCSelectLMHeadModel.from_pretrained(
            args.model_id,
            device=device,
            cache_dir=args.cache_dir,
            segment_size=args.mc_segment_size,
            max_cached_segments=args.mc_max_cached_segments,
            keep_top_k=args.mc_select_keep_top_k,
            score_threshold=args.mc_select_score_threshold,
            detach_cached_segments=(not args.mc_backprop_history),
        )

    ckpt_path = Path(args.checkpoint_dir) / "pytorch_model.bin"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. "
            "If you used the new finetune naming, try a folder like Mamba2-final or Mamba2MC-final."
        )

    state_dict = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Warning: missing keys while loading checkpoint: {missing_keys[:8]}")
    if unexpected_keys:
        print(f"Warning: unexpected keys while loading checkpoint: {unexpected_keys[:8]}")

    model.eval()
    return model


def generate_text(model, tokenizer, args: argparse.Namespace, device: torch.device) -> str:
    torch.manual_seed(args.seed)
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)[0]

    pieces = [args.prompt]
    for token_id, _ in model.generate(
        input_ids,
        max_new_length=args.max_new_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=tokenizer.eos_token_id,
    ):
        pieces.append(tokenizer.decode([token_id]))
    return "".join(pieces)


def _text_observation_metrics(text: str) -> dict:
    words = re.findall(r"[A-Za-z']+", text)
    alpha_chars = sum(1 for ch in text if ch.isalpha())
    printable_chars = sum(1 for ch in text if ch.isprintable())
    repeated_3gram_count = 0

    clean = " ".join(text.split())
    tokens = clean.split()
    if len(tokens) >= 6:
        seen = {}
        for i in range(len(tokens) - 2):
            tri = tuple(tokens[i : i + 3])
            seen[tri] = seen.get(tri, 0) + 1
        repeated_3gram_count = sum(v - 1 for v in seen.values() if v > 1)

    return {
        "length": len(text),
        "word_count": len(words),
        "alpha_ratio": (alpha_chars / max(1, len(text))),
        "printable_ratio": (printable_chars / max(1, len(text))),
        "repeated_3gram_count": repeated_3gram_count,
    }


def sanity_check_checkpoint(model, tokenizer, args: argparse.Namespace, device: torch.device) -> None:
    print("\n=== Checkpoint Sanity Checks ===")
    prompts = [
        "The purpose of this project is",
        "In summary, the model should",
        "Q: What is machine learning?\nA:",
    ]

    looks_reasonable = True
    for idx, prompt in enumerate(prompts, start=1):
        test_args = argparse.Namespace(**vars(args))
        test_args.prompt = prompt
        test_args.max_new_length = min(64, args.max_new_length)
        test_args.temperature = min(args.temperature, 0.9)
        sample = generate_text(model, tokenizer, test_args, device)
        metrics = _text_observation_metrics(sample)

        print(f"\n[Sanity Sample {idx}] prompt: {prompt!r}")
        print(sample[:400].replace("\n", "\\n"))
        print(
            "metrics:",
            f"len={metrics['length']}, words={metrics['word_count']}, "
            f"alpha_ratio={metrics['alpha_ratio']:.2f}, "
            f"printable_ratio={metrics['printable_ratio']:.2f}, "
            f"repeated_3gram_count={metrics['repeated_3gram_count']}",
        )

        if (
            metrics["printable_ratio"] < 0.98
            or metrics["alpha_ratio"] < 0.45
            or metrics["repeated_3gram_count"] > 8
        ):
            looks_reasonable = False

    if looks_reasonable:
        print("Checkpoint sanity result: PASS (outputs look mostly coherent).")
    else:
        print(
            "Checkpoint sanity result: WARNING (some outputs look noisy/repetitive). "
            "Try lowering temperature/top_k or double-check the fine-tuned checkpoint."
        )


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id, cache_dir=args.cache_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = load_model(args, device)
    sanity_check_checkpoint(model, tokenizer, args, device)
    text = generate_text(model, tokenizer, args, device)
    print("\n=== Generated Text ===")
    print(text)


if __name__ == "__main__":
    main()
