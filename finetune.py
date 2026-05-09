import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from checkpoint_metadata import (
    TRAINING_ALIGNMENT_FIELDS,
    align_args_with_checkpoint_geometry,
    collect_training_geometry,
    find_explicit_arg_dests,
    parse_saved_training_geometry,
    print_training_geometry,
    read_checkpoint_meta,
    write_checkpoint_meta,
)

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _wandb_run_name(args) -> str:
    return f"{args.model_type}-freeze{args.freeze_epochs}-finetune{args.full_finetune_epochs}"


def wandb_enabled_from_env() -> bool:
    return bool(os.getenv("WANDB_APIKEY") or os.getenv("WANDB_API_KEY"))


def init_wandb_run(args):
    # Support user-provided key naming: WANDB_APIKEY="..."
    api_key = os.getenv("WANDB_API_KEY") or os.getenv("WANDB_APIKEY")
    if not api_key:
        return None
    os.environ["WANDB_API_KEY"] = api_key

    try:
        import wandb
    except ModuleNotFoundError:
        print("W&B disabled: package 'wandb' is not installed.")
        return None

    run_name = _wandb_run_name(args)
    try:
        run = wandb.init(
            project="567-mamba-finetune",
            name=run_name,
            config={k: v for k, v in vars(args).items() if not k.startswith("_")},
        )
        print(f"W&B initialized: project=567-mamba-finetune run={run_name}")
        return run
    except Exception as exc:
        print(f"Warning: failed to initialize W&B ({exc}). Continuing without W&B.")
        return None


def wandb_log(run: Any, payload: dict, step: int | None = None) -> None:
    if run is None:
        return
    try:
        if step is None:
            run.log(payload)
        else:
            run.log(payload, step=step)
    except Exception as exc:
        print(f"Warning: W&B log failed ({exc}).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Mamba2 with staged freeze->unfreeze training."
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default="Mamba2",
        help="Model Class Name",
    )

    parser.add_argument("--model-id", type=str, default="state-spaces/mamba2-1.3b")
    parser.add_argument("--tokenizer-id", type=str, default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--cache-dir", type=str, default="./huggingface_cache")
    parser.add_argument("--output-dir", type=str, default="./checkpoints/mamba2-finetune-54M")
    parser.add_argument(
        "--train-datasets",
        nargs="+",
        default=["both"],
        choices=["wikitext", "fineweb", "both"],
        help="Pick one, or use 'both' to run WikiText then FineWeb.",
    )
    parser.add_argument(
        "--dataset-strategy",
        type=str,
        default="mix",
        choices=["mix", "sequential"],
        help="mix: combine selected datasets into one shuffled pool per epoch; sequential: train one dataset after another.",
    )

    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=1,
        help="Stage-1 epochs: freeze base weights and train only skeleton params.",
    )
    parser.add_argument(
        "--full-finetune-epochs",
        type=int,
        default=2,
        help="Stage-2 epochs: unfreeze all params and fine-tune end-to-end.",
    )

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.01)
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "linear", "constant"],
        help="LR schedule after warmup.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--block-size",
        type=int,
        default=256,
        help="Should be a multiple of 64. Default 256 gives MC/MCSelect at least 4 segments with mc-segment-size 64.",
    )
    parser.add_argument(
        "--mc-segment-size",
        type=int,
        default=64,
        help="Segment length for Mamba2MC cache boundaries. Use < block-size so MC params affect logits during freeze stage.",
    )
    parser.add_argument(
        "--mc-max-cached-segments",
        type=int,
        default=16,
        help="Keep only the latest X cached segments in Mamba2MC. 0 means keep all.",
    )
    parser.add_argument(
        "--mc-backprop-history",
        action="store_true",
        help="Backpropagate through historical cached segments. Disabled by default for better speed and memory.",
    )
    parser.add_argument(
        "--mc-online-bias-init",
        type=float,
        default=1.0,
        help="Initialization value for Mamba2MC online_bias (recommended 0-2).",
    )
    parser.add_argument(
        "--mc-freeze-train-mode",
        type=str,
        default="mc_plus_norm",
        choices=["mc_only", "mc_plus_norm", "all"],
        help=(
            "Which params to train during Mamba2MC freeze stage: "
            "mc_only trains only MC params; mc_plus_norm also trains norm params; "
            "all trains all params."
        ),
    )
    parser.add_argument(
        "--mc-select-keep-top-k",
        type=int,
        default=8,
        help="For Mamba2MCSelect: keep top-k scored cached segments when cache overflows. 0 means keep all.",
    )
    parser.add_argument(
        "--mc-select-score-threshold",
        type=float,
        default=-1.0,
        help="For Mamba2MCSelect: discard cached segments with score lower than this threshold during pruning.",
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--dataloader-prefetch-factor",
        type=int,
        default=4,
        help="Number of prefetched batches per DataLoader worker.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--debug-first-signal-batches",
        type=int,
        default=0,
        help=(
            "Print per-batch gradient diagnostics for W/selector params on the first N "
            "training batches after backward. 0 disables."
        ),
    )
    parser.add_argument(
        "--log-every-data-points",
        type=int,
        default=200,
        help="Print training loss/ppl every N processed data points.",
    )
    parser.add_argument(
        "--run-initial-validation",
        action="store_true",
        help="Run validation before training updates in each stage/epoch.",
    )
    parser.add_argument(
        "--val-fraction-of-train",
        type=float,
        default=0.10,
        help="Validation subset size as a fraction of train token target.",
    )

    parser.add_argument(
        "--wikitext-config",
        type=str,
        default="wikitext-103-raw-v1",
        help="Config under Salesforce/wikitext.",
    )
    parser.add_argument(
        "--max-train-samples-wikitext",
        type=int,
        default=0,
        help="Debug only. 0 means use full WikiText train split.",
    )
    parser.add_argument(
        "--wikitext-target-tokens",
        type=int,
        default=10_000_000,
        help="Optional token cap for WikiText train text collection. 0 means no token cap.",
    )

    parser.add_argument(
        "--fineweb-config",
        type=str,
        default="CC-MAIN-2024-10",
        help="FineWeb subset/config to stream.",
    )
    parser.add_argument(
        "--fineweb-target-tokens",
        type=int,
        default=17_000_000,
        help="Target number of tokenizer tokens to collect from FineWeb train stream.",
    )
    parser.add_argument(
        "--fineweb-max-stream-samples",
        type=int,
        default=0,
        help="Optional safety cap on number of streamed docs; 0 disables.",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory for periodic data-point checkpoints.",
    )
    parser.add_argument(
        "--data-point-checkpoint-interval",
        type=int,
        default=10000,
        help="Save a checkpoint every N processed data points.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default="",
        help="Path to a checkpoint directory containing pytorch_model.bin (and optional meta.txt/tokenizer files).",
    )
    args = parser.parse_args()
    args._explicit_dests = find_explicit_arg_dests(parser, sys.argv[1:])
    return args


def _training_checkpoint_meta(
    args,
    *,
    stage_tag: str | None = None,
    epoch_idx: int | None = None,
    data_points: int | None = None,
    global_step: int | None = None,
) -> dict[str, Any]:
    meta = collect_training_geometry(args)
    if stage_tag is not None:
        meta["stage"] = stage_tag
    if epoch_idx is not None:
        meta["epoch"] = epoch_idx
    if data_points is not None:
        meta["data_points"] = data_points
    if global_step is not None:
        meta["global_step"] = global_step
    return meta


def _mc_segments_per_sample(args) -> int:
    return args.block_size // args.mc_segment_size


def _mc_training_has_selector_signal(args) -> bool:
    return args.model_type in {"Mamba2MC", "Mamba2MCSelect"} and _mc_segments_per_sample(args) >= 3


def _print_effective_training_geometry(args) -> None:
    print("\n=== Training Geometry ===")
    print(f"model_type: {args.model_type}")
    print(f"block_size: {args.block_size}")
    if args.model_type not in {"Mamba2MC", "Mamba2MCSelect"}:
        return

    segments_per_sample = _mc_segments_per_sample(args)
    print(f"mc_segment_size: {args.mc_segment_size}")
    print(f"segments_per_sample: {segments_per_sample}")
    print(f"mc_max_cached_segments: {args.mc_max_cached_segments}")
    if args.model_type == "Mamba2MCSelect":
        print(f"mc_select_keep_top_k: {args.mc_select_keep_top_k}")
        print(f"mc_select_score_threshold: {args.mc_select_score_threshold}")
        print(
            "Signal note: online_bias affects the second segment; "
            "W, select_score, and select_score_mix_weight get nontrivial signal once a third segment exists."
        )
    else:
        print(
            "Signal note: online_bias affects the second segment; "
            "W gets nontrivial signal once a third segment exists."
        )


def _validate_training_geometry(args) -> None:
    if args.model_type not in {"Mamba2MC", "Mamba2MCSelect"}:
        return

    segments_per_sample = _mc_segments_per_sample(args)
    if segments_per_sample < 3:
        raise ValueError(
            "Mamba2MC/Mamba2MCSelect need at least 3 full segments per packed training sample "
            "to give W/select_score a real training signal. "
            f"Current geometry: block_size={args.block_size}, mc_segment_size={args.mc_segment_size}, "
            f"segments_per_sample={segments_per_sample}. "
            "Use a larger --block-size (recommended: --block-size 256 with --mc-segment-size 64)."
        )


def _signal_param_names_for_model(model) -> tuple[str, ...]:
    model_name = model.__class__.__name__
    if model_name == "Mamba2MCLMHeadModel":
        return ("W", "online_bias")
    if model_name == "Mamba2MCSelectLMHeadModel":
        return (
            "W",
            "online_bias",
            "select_score.weight",
            "select_score.bias",
            "select_score_mix_weight",
        )
    return ()


def _verify_mc_checkpoint_keys(model) -> None:
    required = _signal_param_names_for_model(model)
    if not required:
        return
    state_dict = model.state_dict()
    missing = [name for name in required if name not in state_dict]
    if missing:
        raise RuntimeError(
            f"Refusing to save checkpoint because expected MC params are missing from state_dict: {missing}"
        )


def _print_signal_param_trainability(model) -> None:
    param_names = _signal_param_names_for_model(model)
    if not param_names:
        return
    named_params = dict(model.named_parameters())
    state_dict = model.state_dict()
    print("Signal parameter status:")
    for name in param_names:
        param = named_params.get(name)
        print(
            f"  {name}: in_state_dict={name in state_dict}, "
            f"requires_grad={param.requires_grad if param is not None else False}"
        )


def _new_signal_tracker(param_names: tuple[str, ...]) -> dict[str, float]:
    return {name: 0.0 for name in param_names}


def _accumulate_signal_grad_norms(model, tracker: dict[str, float]) -> None:
    if not tracker:
        return
    named_params = dict(model.named_parameters())
    for name in tracker:
        param = named_params.get(name)
        if param is None or param.grad is None:
            continue
        grad_norm = float(param.grad.detach().float().norm().item())
        if grad_norm > tracker[name]:
            tracker[name] = grad_norm


def _log_signal_diagnostics(
    model,
    tracker: dict[str, float],
    *,
    stage_tag: str,
    epoch_idx: int,
    global_step: int,
    data_points_seen: int,
    wandb_run=None,
    interval_tag: str = "interval",
) -> None:
    if not tracker:
        return

    named_params = dict(model.named_parameters())
    parts = []
    payload: dict[str, float | int | str] = {
        "signal/stage": stage_tag,
        "signal/epoch": epoch_idx,
        "signal/data_points": data_points_seen,
    }
    for name, max_grad_norm in tracker.items():
        param = named_params.get(name)
        if param is None:
            continue
        param_norm = float(param.detach().float().norm().item())
        safe_name = name.replace(".", "_")
        parts.append(
            f"{name}: norm={param_norm:.6f}, max_grad_norm={max_grad_norm:.6e}"
        )
        payload[f"signal/{safe_name}/norm"] = param_norm
        payload[f"signal/{safe_name}/max_grad_norm"] = max_grad_norm

    if parts:
        print(
            f"Signal diagnostics ({interval_tag}) | stage={stage_tag} epoch={epoch_idx} "
            f"step={global_step} data_points={data_points_seen} | "
            + " | ".join(parts)
        )
        wandb_log(wandb_run, payload, step=global_step)


def _warn_for_missing_signal(
    model,
    tracker: dict[str, float],
    args,
) -> None:
    if not tracker or not _mc_training_has_selector_signal(args):
        return

    model_name = model.__class__.__name__
    if model_name == "Mamba2MCSelectLMHeadModel":
        if tracker.get("select_score.weight", 0.0) == 0.0 and tracker.get("select_score.bias", 0.0) == 0.0:
            print(
                "Warning: select_score gradients stayed at zero for the first full logging interval. "
                "The selector head is likely not receiving a useful training signal."
            )
    elif model_name == "Mamba2MCLMHeadModel" and tracker.get("W", 0.0) == 0.0:
        print(
            "Warning: W gradients stayed at zero for the first full logging interval. "
            "The MC history mixer is likely not receiving a useful training signal."
        )


def _print_step_signal_debug(
    model,
    *,
    stage_tag: str,
    dataset_tag: str,
    epoch_idx: int,
    batch_idx: int,
    data_points_seen: int,
    loss_value: float,
) -> None:
    param_names = _signal_param_names_for_model(model)
    if not param_names:
        return

    named_params = dict(model.named_parameters())
    parts = []
    for name in param_names:
        param = named_params.get(name)
        if param is None:
            parts.append(f"{name}: missing")
            continue
        grad = param.grad
        if grad is None:
            parts.append(f"{name}: grad=None")
            continue
        grad_norm = float(grad.detach().float().norm().item())
        grad_max = float(grad.detach().float().abs().max().item())
        parts.append(f"{name}: grad_norm={grad_norm:.6e}, grad_abs_max={grad_max:.6e}")

    print(
        f"Signal debug | stage={stage_tag} dataset={dataset_tag} epoch={epoch_idx} "
        f"batch={batch_idx} data_points={data_points_seen} loss={loss_value:.6f} | "
        + " | ".join(parts)
    )


def _save_training_bundle(
    ckpt_dir: Path,
    *,
    optimizer: AdamW | None = None,
    scheduler: LambdaLR | None = None,
    scheduler_config: dict[str, Any] | None = None,
    stage_tag: str | None = None,
    epoch_idx: int | None = None,
    dataset_tag: str | None = None,
    batch_step: int | None = None,
    global_step: int | None = None,
    data_points_seen: int | None = None,
    next_data_point_checkpoint: int | None = None,
) -> None:
    payload: dict[str, Any] = {
        "stage_tag": stage_tag,
        "epoch_idx": epoch_idx,
        "dataset_tag": dataset_tag,
        "batch_step": batch_step,
        "global_step": global_step,
        "data_points_seen": data_points_seen,
        "next_data_point_checkpoint": next_data_point_checkpoint,
        "scheduler_config": scheduler_config,
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, ckpt_dir / "training_state.pt")


def _load_training_bundle(ckpt_dir: Path):
    bundle_path = ckpt_dir / "training_state.pt"
    if not bundle_path.exists():
        return None
    return torch.load(bundle_path, map_location="cpu", weights_only=False)


def _move_optimizer_state_to_device(optimizer: AdamW, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _restore_rng_state_from_bundle(resume_bundle) -> None:
    if not resume_bundle:
        return
    torch_rng_state = resume_bundle.get("torch_rng_state")
    if torch_rng_state is not None:
        torch.set_rng_state(torch_rng_state)
    cuda_rng_state_all = resume_bundle.get("cuda_rng_state_all")
    if cuda_rng_state_all is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_rng_state_all)


def _resume_bundle_matches_stage(resume_bundle, stage_tag: str) -> bool:
    return bool(resume_bundle) and resume_bundle.get("stage_tag") == stage_tag


def tokenize_and_pack_text_dataset(text_dataset: Dataset, tokenizer, block_size: int) -> Dataset:
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must have eos_token_id for document boundary packing.")

    def tokenize_fn(examples):
        texts = [t for t in examples["text"] if t and not t.isspace()]
        tokenized = tokenizer(texts, return_attention_mask=False)
        input_ids = []
        for seq in tokenized["input_ids"]:
            # Insert EOS between docs to preserve boundaries during packing.
            input_ids.append(seq + [eos_token_id])
        return {"input_ids": input_ids}

    def group_texts(examples):
        concatenated = []
        for seq in examples["input_ids"]:
            concatenated.extend(seq)

        total_len = (len(concatenated) // block_size) * block_size
        concatenated = concatenated[:total_len]

        input_ids = [
            concatenated[i : i + block_size]
            for i in range(0, total_len, block_size)
        ]
        labels = [chunk.copy() for chunk in input_ids]
        return {"input_ids": input_ids, "labels": labels}

    tokenized = text_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=text_dataset.column_names,
        desc="Tokenizing text",
    )
    packed = tokenized.map(group_texts, batched=True, desc="Packing token blocks")
    packed.set_format(type="torch", columns=["input_ids", "labels"])
    return packed


def build_dataloader(dataset: Dataset, args, shuffle: bool) -> DataLoader:
    use_cuda = torch.cuda.is_available()
    kwargs = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "num_workers": args.dataloader_num_workers,
        "pin_memory": use_cuda,
    }
    if args.dataloader_num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = args.dataloader_prefetch_factor
    return DataLoader(dataset, **kwargs)


def _collection_cache_root(args) -> Path:
    root = Path(args.cache_dir) / "text_collection_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _collection_cache_key(spec: dict) -> str:
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _collection_cache_dir(args, spec: dict) -> Path:
    source = str(spec.get("source", "dataset"))
    split = str(spec.get("split", "train"))
    key = _collection_cache_key(spec)
    return _collection_cache_root(args) / f"{source}-{split}-{key}"


def _load_cached_text_collection(args, spec: dict) -> Dataset | None:
    cache_dir = _collection_cache_dir(args, spec)
    meta_path = cache_dir / "meta.json"
    data_dir = cache_dir / "dataset"
    if not meta_path.exists() or not data_dir.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("spec") != spec:
            return None
        dataset = load_from_disk(str(data_dir))
        print(f"Loaded cached text collection from {cache_dir}")
        return dataset
    except Exception as exc:
        print(f"Warning: failed to load cached text collection at {cache_dir}: {exc}")
        return None


def _save_cached_text_collection(args, spec: dict, dataset: Dataset) -> None:
    cache_dir = _collection_cache_dir(args, spec)
    data_dir = cache_dir / "dataset"
    meta_path = cache_dir / "meta.json"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if data_dir.exists() and meta_path.exists():
        print(f"Text collection cache already exists at {cache_dir}")
        return
    dataset.save_to_disk(str(data_dir))
    meta = {
        "spec": spec,
        "num_rows": len(dataset),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved text collection cache to {cache_dir}")


def _collect_texts_by_token_target(
    text_iterable,
    tokenizer,
    target_tokens: int,
    max_stream_samples: int = 0,
    source_name: str = "dataset",
):
    texts = []
    total_tokens = 0
    total_items = None
    progress_mode = "tokens" if target_tokens > 0 else "examples"
    if target_tokens > 0:
        total_items = target_tokens
    elif max_stream_samples > 0:
        total_items = max_stream_samples
    else:
        try:
            total_items = len(text_iterable)
        except TypeError:
            total_items = None

    print(f"Collecting {source_name}: start")
    next_percent_to_print = 1
    last_percent_printed = 0
    next_checkpoint_docs = 10_000
    processed_docs = 0

    def maybe_print_percent():
        nonlocal next_percent_to_print, last_percent_printed
        if total_items is None or total_items <= 0:
            return
        progress_value = total_tokens if progress_mode == "tokens" else processed_docs
        progress_pct = int((100 * progress_value) / total_items)
        while next_percent_to_print <= 100 and progress_pct >= next_percent_to_print:
            print(
                f"Collecting {source_name}: {next_percent_to_print}% "
                f"(docs={processed_docs}, tokens={total_tokens})"
            )
            last_percent_printed = next_percent_to_print
            next_percent_to_print += 1

    for i, sample in enumerate(text_iterable):
        processed_docs += 1
        text = sample.get("text", "")
        if not text or text.isspace():
            maybe_print_percent()
            continue

        token_count = len(tokenizer.encode(text, add_special_tokens=False))
        if token_count == 0:
            maybe_print_percent()
            continue

        if target_tokens > 0 and (total_tokens + token_count) > target_tokens:
            break

        texts.append(text)
        total_tokens += token_count
        maybe_print_percent()

        if total_items is None and processed_docs >= next_checkpoint_docs:
            print(
                f"Collecting {source_name}: docs={processed_docs}, tokens={total_tokens}"
            )
            next_checkpoint_docs += 10_000

        if max_stream_samples > 0 and (i + 1) >= max_stream_samples:
            break

    completed = False
    if progress_mode == "tokens" and target_tokens > 0:
        completed = total_tokens >= target_tokens
    elif total_items is not None and total_items > 0:
        completed = processed_docs >= total_items

    if completed and total_items is not None and total_items > 0 and last_percent_printed < 100:
        print(
            f"Collecting {source_name}: 100% "
            f"(docs={processed_docs}, tokens={total_tokens})"
        )
    else:
        print(f"Collecting {source_name}: done (docs={processed_docs}, tokens={total_tokens})")
    return texts, total_tokens


def _build_wikitext_train_text_dataset(args, tokenizer) -> Dataset:
    spec = {
        "source": "wikitext",
        "split": "train",
        "wikitext_config": args.wikitext_config,
        "max_train_samples_wikitext": args.max_train_samples_wikitext,
        "wikitext_target_tokens": args.wikitext_target_tokens,
        "tokenizer_id": args.tokenizer_id,
    }
    cached_dataset = _load_cached_text_collection(args, spec)
    if cached_dataset is not None:
        return cached_dataset

    print("Loading WikiText train split...")
    dataset = load_dataset("Salesforce/wikitext", args.wikitext_config, split="train", cache_dir=args.cache_dir)

    if args.max_train_samples_wikitext > 0:
        dataset = dataset.select(range(min(args.max_train_samples_wikitext, len(dataset))))

    if args.wikitext_target_tokens > 0:
        print("Collecting WikiText documents to token target...")
        texts, total_tokens = _collect_texts_by_token_target(
            dataset,
            tokenizer=tokenizer,
            target_tokens=args.wikitext_target_tokens,
            source_name="wikitext",
        )
        if not texts:
            raise RuntimeError("No WikiText text was collected under --wikitext-target-tokens.")
        print(
            f"Collected WikiText slice: docs={len(texts)}, tokens={total_tokens} "
            f"(cap={args.wikitext_target_tokens})"
        )
        dataset = Dataset.from_dict({"text": texts})

    _save_cached_text_collection(args, spec, dataset)
    return dataset


def _build_fineweb_train_text_dataset(args, tokenizer) -> Dataset:
    spec = {
        "source": "fineweb",
        "split": "train",
        "fineweb_config": args.fineweb_config,
        "fineweb_target_tokens": args.fineweb_target_tokens,
        "fineweb_max_stream_samples": args.fineweb_max_stream_samples,
        "tokenizer_id": args.tokenizer_id,
    }
    cached_dataset = _load_cached_text_collection(args, spec)
    if cached_dataset is not None:
        return cached_dataset

    print("Loading FineWeb train stream...")
    streamed = load_dataset(
        "HuggingFaceFW/fineweb",
        name=args.fineweb_config,
        split="train",
        streaming=True,
        cache_dir=args.cache_dir,
    )

    print("Collecting FineWeb documents to token target...")
    texts, total_tokens = _collect_texts_by_token_target(
        streamed,
        tokenizer=tokenizer,
        target_tokens=args.fineweb_target_tokens,
        max_stream_samples=args.fineweb_max_stream_samples,
        source_name="fineweb",
    )

    if not texts:
        raise RuntimeError("No FineWeb text was collected. Try another --fineweb-config or token target.")

    print(
        f"Collected FineWeb slice: docs={len(texts)}, tokens={total_tokens} "
        f"(cap={args.fineweb_target_tokens})"
    )
    dataset = Dataset.from_dict({"text": texts})
    _save_cached_text_collection(args, spec, dataset)
    return dataset


def get_wikitext_loader(args, tokenizer) -> DataLoader:
    dataset = _build_wikitext_train_text_dataset(args, tokenizer)
    print("Tokenizing and packing WikiText...")
    packed = tokenize_and_pack_text_dataset(dataset, tokenizer, args.block_size)
    return build_dataloader(packed, args, shuffle=True)


def get_wikitext_text_dataset(args) -> Dataset:
    dataset = load_dataset("Salesforce/wikitext", args.wikitext_config, split="train", cache_dir=args.cache_dir)
    if args.max_train_samples_wikitext > 0:
        dataset = dataset.select(range(min(args.max_train_samples_wikitext, len(dataset))))
    if args.wikitext_target_tokens > 0:
        raise ValueError("wikitext token targeting requires tokenizer; use get_wikitext_loader or get_mixed_loader.")
    return dataset


def get_fineweb_loader(args, tokenizer) -> DataLoader:
    text_dataset = _build_fineweb_train_text_dataset(args, tokenizer)
    print("Tokenizing and packing FineWeb...")
    packed = tokenize_and_pack_text_dataset(text_dataset, tokenizer, args.block_size)
    return build_dataloader(packed, args, shuffle=True)


def get_fineweb_text_dataset(args) -> Dataset:
    raise ValueError("fineweb token targeting requires tokenizer; use get_fineweb_loader or get_mixed_loader.")


def get_wikitext_val_loader(args, tokenizer) -> DataLoader:
    dataset = load_dataset("Salesforce/wikitext", args.wikitext_config, split="validation", cache_dir=args.cache_dir)
    if args.wikitext_target_tokens > 0:
        val_target_tokens = max(1, int(args.wikitext_target_tokens * args.val_fraction_of_train))
        texts, total_tokens = _collect_texts_by_token_target(
            dataset,
            tokenizer=tokenizer,
            target_tokens=val_target_tokens,
            source_name="wikitext-val",
        )
        if not texts:
            raise RuntimeError("No WikiText validation text was collected.")
        print(
            f"Collected WikiText validation slice: docs={len(texts)}, tokens={total_tokens} "
            f"(target~{val_target_tokens})"
        )
        dataset = Dataset.from_dict({"text": texts})
    packed = tokenize_and_pack_text_dataset(dataset, tokenizer, args.block_size)
    return build_dataloader(packed, args, shuffle=False)


def get_fineweb_val_loader(args, tokenizer) -> DataLoader:
    val_target_tokens = max(1, int(args.fineweb_target_tokens * args.val_fraction_of_train))
    streamed = load_dataset(
        "HuggingFaceFW/fineweb",
        name=args.fineweb_config,
        split="train",
        streaming=True,
        cache_dir=args.cache_dir,
    )

    texts, total_tokens = _collect_texts_by_token_target(
        streamed,
        tokenizer=tokenizer,
        target_tokens=val_target_tokens,
        source_name="fineweb-val",
    )
    if not texts:
        raise RuntimeError("No FineWeb validation text was collected.")
    print(
        f"Collected FineWeb validation slice: docs={len(texts)}, tokens={total_tokens} "
        f"(target~{val_target_tokens})"
    )

    text_dataset = Dataset.from_dict({"text": texts})
    packed = tokenize_and_pack_text_dataset(text_dataset, tokenizer, args.block_size)
    return build_dataloader(packed, args, shuffle=False)


def get_mixed_loader(args, tokenizer, selected_datasets: list[str]) -> DataLoader:
    text_datasets = []

    if "wikitext" in selected_datasets:
        text_datasets.append(_build_wikitext_train_text_dataset(args, tokenizer))
    if "fineweb" in selected_datasets:
        text_datasets.append(_build_fineweb_train_text_dataset(args, tokenizer))

    if not text_datasets:
        raise ValueError("No datasets selected for mixed training.")

    if len(text_datasets) == 1:
        mixed_text_dataset = text_datasets[0]
    else:
        mixed_text_dataset = concatenate_datasets(text_datasets)

    print("Tokenizing and packing mixed dataset...")
    packed = tokenize_and_pack_text_dataset(mixed_text_dataset, tokenizer, args.block_size)
    return build_dataloader(packed, args, shuffle=True)


def normalize_dataset_choice(train_datasets: list[str]) -> list[str]:
    if "both" in train_datasets:
        return ["wikitext", "fineweb"]
    ordered = []
    for name in train_datasets:
        if name not in ordered:
            ordered.append(name)
    return ordered


def _is_mc_specific_trainable_param(name: str) -> bool:
    """Return True for Mamba2MC-only params intended for freeze-stage training."""
    return (
        name == "W"
        or name.endswith(".W")
        or "mc_weight_matrix" in name
        or "online_bias" in name
        or "select_score" in name
    )


def _is_norm_param(name: str) -> bool:
    return "norm" in name.lower()


def configure_stage1_trainability_by_model_name(model, args) -> tuple[str, list[str], list[str]]:
    model_name = model.__class__.__name__

    if model_name == "Mamba2LMHeadModel":
        # Current request: for the current Mamba2 model, keep all params trainable.
        for _, param in model.named_parameters():
            param.requires_grad = True
    elif model_name in {"Mamba2MCLMHeadModel", "Mamba2MCSelectLMHeadModel"}:
        # Freeze-stage policy for MC model is configurable.
        for name, param in model.named_parameters():
            if args.mc_freeze_train_mode == "mc_only":
                param.requires_grad = _is_mc_specific_trainable_param(name)
            elif args.mc_freeze_train_mode == "mc_plus_norm":
                param.requires_grad = _is_mc_specific_trainable_param(name) or _is_norm_param(name)
            elif args.mc_freeze_train_mode == "all":
                param.requires_grad = True
            else:
                param.requires_grad = _is_mc_specific_trainable_param(name)
    else:
        # pass for all other model names: leave requires_grad unchanged.
        pass

    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    frozen_names = [name for name, param in model.named_parameters() if not param.requires_grad]
    return model_name, trainable_names, frozen_names


def load_mamba2_model(model_id: str, device, cache_dir: str):
    try:
        from mamba2 import Mamba2LMHeadModel
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Failed to import mamba2 dependencies. Install requirements first: pip install -r requirements.txt"
        ) from exc

    return Mamba2LMHeadModel.from_pretrained(model_id, device=device, cache_dir=cache_dir)

def load_our_model(
    model_id: str,
    device,
    cache_dir: str,
    segment_size: int,
    max_cached_segments: int,
    detach_cached_segments: bool,
):
    try:
        from mamba2_mc import Mamba2MCLMHeadModel
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Failed to import mamba2_mc dependencies. Install requirements first: pip install -r requirements.txt"
        ) from exc

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
    try:
        from mamba2_mc_select import Mamba2MCSelectLMHeadModel
    except ModuleNotFoundError as exc:
        try:
            from mc_select import Mamba2MCSelectLMHeadModel
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Failed to import mamba2_mc_select dependencies. Install requirements first: pip install -r requirements.txt"
            ) from exc

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

def unfreeze_all_params(model) -> None:
    for _, param in model.named_parameters():
        param.requires_grad = True


def build_optimizer(model, args) -> AdamW:
    named_trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not named_trainable_params:
        raise ValueError("No trainable parameters found for optimizer.")
    no_decay_params = []
    decay_params = []
    for name, param in named_trainable_params:
        lname = name.lower()
        if (
            lname.endswith("bias")
            or "norm" in lname
            or "embedding" in lname
            or lname.endswith("online_bias")
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": args.weight_decay})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return AdamW(param_groups, lr=args.learning_rate)


def build_warmup_decay_scheduler(
    optimizer, warmup_steps: int, total_steps: int, scheduler_name: str
) -> LambdaLR:
    if total_steps <= 0:
        return LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    warmup_steps = max(0, min(warmup_steps, total_steps))

    def lr_lambda(step: int) -> float:
        current_step = step + 1
        if warmup_steps > 0 and current_step <= warmup_steps:
            return float(current_step) / float(warmup_steps)
        if scheduler_name == "constant":
            return 1.0
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, float(current_step - warmup_steps) / float(decay_steps)))
        if scheduler_name == "linear":
            return max(0.0, 1.0 - progress)
        if scheduler_name == "cosine":
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return 1.0

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def perplexity_from_loss(loss_value: float) -> float:
    # Use the true perplexity from NLL loss; no artificial clipping.
    try:
        return math.exp(loss_value)
    except OverflowError:
        return float("inf")


def save_data_point_checkpoint(
    model,
    tokenizer,
    args,
    optimizer,
    scheduler,
    scheduler_config,
    checkpoint_dir: str,
    data_points_mark: int,
    stage_tag: str,
    epoch_idx: int,
    dataset_tag: str,
    batch_step: int,
    model_type: str,
    global_step: int,
    next_data_point_checkpoint: int,
) -> None:
    ckpt_dir = Path(checkpoint_dir) / f"{model_type}-data-points-{data_points_mark}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _verify_mc_checkpoint_keys(model)
    torch.save(model.state_dict(), ckpt_dir / "pytorch_model.bin")
    tokenizer.save_pretrained(ckpt_dir)
    write_checkpoint_meta(
        ckpt_dir / "meta.txt",
        _training_checkpoint_meta(
            args,
            stage_tag=stage_tag,
            epoch_idx=epoch_idx,
            data_points=data_points_mark,
            global_step=global_step,
        ),
    )
    _save_training_bundle(
        ckpt_dir,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_config=scheduler_config,
        stage_tag=stage_tag,
        epoch_idx=epoch_idx,
        dataset_tag=dataset_tag,
        batch_step=batch_step,
        global_step=global_step,
        data_points_seen=data_points_mark,
        next_data_point_checkpoint=next_data_point_checkpoint,
    )
    print(f"Saved data-point checkpoint to {ckpt_dir}")


def _read_checkpoint_meta(ckpt_dir: Path) -> dict[str, str]:
    return read_checkpoint_meta(ckpt_dir)


def _parse_checkpoint_stage_epoch_from_dirname(ckpt_dir: Path) -> tuple[str | None, int | None]:
    name = ckpt_dir.name
    match = re.search(r"-(freeze|full)-epoch-(\d+)-", name)
    if not match:
        return None, None
    stage = match.group(1)
    try:
        epoch = int(match.group(2))
    except ValueError:
        return stage, None
    return stage, epoch


def _resolve_resume_stage_epoch(ckpt_dir: Path, ckpt_meta: dict[str, str]) -> tuple[str | None, int]:
    resume_stage = ckpt_meta.get("stage")
    resume_epoch = 0

    if "epoch" in ckpt_meta:
        try:
            resume_epoch = int(ckpt_meta["epoch"])
        except ValueError:
            print(f"Warning: invalid epoch in checkpoint meta: {ckpt_meta['epoch']}")

    if resume_stage is None:
        parsed_stage, parsed_epoch = _parse_checkpoint_stage_epoch_from_dirname(ckpt_dir)
        resume_stage = parsed_stage
        if parsed_epoch is not None:
            resume_epoch = parsed_epoch

    return resume_stage, resume_epoch


def _checkpoint_is_data_point_ckpt(ckpt_dir: Path) -> bool:
    return "-data-points-" in ckpt_dir.name


def _is_mc_model(model) -> bool:
    return model.__class__.__name__ in {"Mamba2MCLMHeadModel", "Mamba2MCSelectLMHeadModel"}


def _is_mc_only_trainable(model) -> bool:
    if not _is_mc_model(model):
        return False
    for name, param in model.named_parameters():
        if param.requires_grad and not _is_mc_specific_trainable_param(name):
            return False
    return True


def compute_batch_loss(model, input_ids, labels) -> torch.Tensor:
    """Compute autoregressive CE loss from full-sequence logits for all model types."""
    logits, _ = model(input_ids)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def train_one_epoch(
    model,
    tokenizer,
    train_loader,
    optimizer,
    device,
    args,
    global_step: int,
    data_points_seen: int,
    next_data_point_checkpoint: int,
    dataset_tag: str,
    stage_tag: str,
    epoch_idx: int,
    stage_epochs: int,
    scheduler=None,
    scheduler_config: dict[str, Any] | None = None,
    val_loader=None,
    val_dataset_tag: str | None = None,
    val_loaders: list[tuple[str, DataLoader]] | None = None,
    wandb_run=None,
) -> tuple[int, int, int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    num_batches = len(train_loader)
    updates_this_epoch = max(1, math.ceil(num_batches / args.grad_accum_steps))
    running_loss = torch.zeros((), device=device)
    running_loss_batches = 0
    next_train_log_checkpoint = None
    if args.log_every_data_points > 0:
        next_train_log_checkpoint = (
            (data_points_seen // args.log_every_data_points) + 1
        ) * args.log_every_data_points
    val_interval = args.data_point_checkpoint_interval if args.data_point_checkpoint_interval > 0 else args.log_every_data_points
    has_validation = (val_loader is not None) or (val_loaders is not None and len(val_loaders) > 0)
    next_val_checkpoint = (
        ((data_points_seen // val_interval) + 1) * val_interval
        if (has_validation and val_interval > 0)
        else None
    )
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    signal_param_names = _signal_param_names_for_model(model)
    signal_grad_tracker = _new_signal_tracker(signal_param_names)
    warned_about_missing_signal = False
    debug_signal_batches_printed = 0

    print(
        f"Training 1 epoch | stage={stage_tag} dataset={dataset_tag}: batches={num_batches}, "
        f"optimizer_updates={updates_this_epoch}"
    )

    progress = tqdm(
        enumerate(train_loader, start=1),
        total=num_batches,
        desc=f"{stage_tag} {epoch_idx}/{stage_epochs} | {dataset_tag}",
        leave=False,
    )

    for step, batch in progress:
        input_ids = batch["input_ids"].to(device, non_blocking=(device.type == "cuda"))
        labels = batch["labels"].to(device, non_blocking=(device.type == "cuda"))
        batch_data_points = int(input_ids.size(0))
        data_points_seen += batch_data_points

        loss = compute_batch_loss(model, input_ids, labels)
        if not loss.requires_grad:
            raise RuntimeError(
                "Loss has no grad_fn. In freeze stage, this usually means trainable MC params "
                "did not influence logits. Try setting --mc-segment-size smaller than --block-size "
                "(for example, --mc-segment-size 64 with --block-size 256), or unfreeze more params."
            )
        loss = loss / args.grad_accum_steps
        loss.backward()
        _accumulate_signal_grad_norms(model, signal_grad_tracker)
        if (
            args.debug_first_signal_batches > 0
            and debug_signal_batches_printed < args.debug_first_signal_batches
            and signal_param_names
        ):
            _print_step_signal_debug(
                model,
                stage_tag=stage_tag,
                dataset_tag=dataset_tag,
                epoch_idx=epoch_idx,
                batch_idx=step,
                data_points_seen=data_points_seen,
                loss_value=float((loss.detach() * args.grad_accum_steps).item()),
            )
            debug_signal_batches_printed += 1
        # Keep logging in true per-batch loss units (not grad-accum scaled units).
        running_loss = running_loss + (loss.detach() * args.grad_accum_steps)
        running_loss_batches += 1

        should_update = (step % args.grad_accum_steps == 0) or (step == num_batches)
        if should_update:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        if (
            next_train_log_checkpoint is not None
            and data_points_seen >= next_train_log_checkpoint
            and running_loss_batches > 0
        ):
            avg_loss = float((running_loss / running_loss_batches).item())
            ppl = perplexity_from_loss(avg_loss)
            if scheduler is not None:
                lr = scheduler.get_last_lr()[0]
            else:
                lr = optimizer.param_groups[0]["lr"]
            print(
                f"stage={stage_tag} dataset={dataset_tag} step={global_step} "
                f"data_points={data_points_seen} loss={avg_loss:.4f} "
                f"ppl={ppl:.3e} lr={lr:.2e}"
            )
            wandb_log(
                wandb_run,
                {
                    "train/loss": avg_loss,
                    "train/ppl": ppl,
                    "train/lr": lr,
                    "train/data_points": data_points_seen,
                    "train/stage": stage_tag,
                    "train/dataset": dataset_tag,
                    "train/epoch": epoch_idx,
                },
                step=global_step,
            )
            _log_signal_diagnostics(
                model,
                signal_grad_tracker,
                stage_tag=stage_tag,
                epoch_idx=epoch_idx,
                global_step=global_step,
                data_points_seen=data_points_seen,
                wandb_run=wandb_run,
            )
            if not warned_about_missing_signal:
                _warn_for_missing_signal(model, signal_grad_tracker, args)
                warned_about_missing_signal = True
            progress.set_postfix_str(f"loss={avg_loss:.4f}")
            running_loss = torch.zeros((), device=device)
            running_loss_batches = 0
            signal_grad_tracker = _new_signal_tracker(signal_param_names)
            while data_points_seen >= next_train_log_checkpoint:
                next_train_log_checkpoint += args.log_every_data_points

        if args.data_point_checkpoint_interval > 0:
            while data_points_seen >= next_data_point_checkpoint:
                save_data_point_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    args=args,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scheduler_config=scheduler_config,
                    checkpoint_dir=args.checkpoint_dir,
                    data_points_mark=next_data_point_checkpoint,
                    stage_tag=stage_tag,
                    epoch_idx=epoch_idx,
                    dataset_tag=dataset_tag,
                    batch_step=step,
                    model_type=args.model_type,
                    global_step=global_step,
                    next_data_point_checkpoint=next_data_point_checkpoint + args.data_point_checkpoint_interval,
                )
                next_data_point_checkpoint += args.data_point_checkpoint_interval

        # Run validation each checkpoint/interval.
        if has_validation and next_val_checkpoint is not None and data_points_seen >= next_val_checkpoint:
            active_val_loaders = val_loaders
            if active_val_loaders is None and val_loader is not None:
                active_val_loaders = [(val_dataset_tag or dataset_tag, val_loader)]
            for eval_name, eval_loader in (active_val_loaders or []):
                val_loss = evaluate(model, eval_loader, device, args)
                val_ppl = perplexity_from_loss(val_loss)
                print(
                    f"Validation | dataset={eval_name} data_points={data_points_seen} "
                    f"loss={val_loss:.4f} ppl={val_ppl:.3e}"
                )
                val_key = re.sub(r"[^A-Za-z0-9_.-]", "_", str(eval_name))
                wandb_log(
                    wandb_run,
                    {
                        f"val/{val_key}/loss": val_loss,
                        f"val/{val_key}/ppl": val_ppl,
                        "val/data_points": data_points_seen,
                        "val/stage": stage_tag,
                        "val/epoch": epoch_idx,
                    },
                    step=global_step,
                )
            while data_points_seen >= next_val_checkpoint:
                next_val_checkpoint += val_interval
            model.train()

    if signal_param_names and (running_loss_batches > 0 or any(value > 0.0 for value in signal_grad_tracker.values())):
        _log_signal_diagnostics(
            model,
            signal_grad_tracker,
            stage_tag=stage_tag,
            epoch_idx=epoch_idx,
            global_step=global_step,
            data_points_seen=data_points_seen,
            wandb_run=wandb_run,
            interval_tag="epoch_tail",
        )
        if not warned_about_missing_signal:
            _warn_for_missing_signal(model, signal_grad_tracker, args)

    return global_step, data_points_seen, next_data_point_checkpoint


def finetune_one_epoch_on_wikitext(
    model,
    tokenizer,
    optimizer,
    device,
    args,
    global_step: int,
    data_points_seen: int,
    next_data_point_checkpoint: int,
    stage_tag: str,
    epoch_idx: int,
    stage_epochs: int,
    scheduler=None,
) -> tuple[int, int, int]:
    loader = get_wikitext_loader(args, tokenizer)
    return train_one_epoch(
        model,
        tokenizer,
        loader,
        optimizer,
        device,
        args,
        global_step,
        data_points_seen,
        next_data_point_checkpoint,
        dataset_tag="wikitext",
        stage_tag=stage_tag,
        epoch_idx=epoch_idx,
        stage_epochs=stage_epochs,
        scheduler=scheduler,
    )


def finetune_one_epoch_on_fineweb(
    model,
    tokenizer,
    optimizer,
    device,
    args,
    global_step: int,
    data_points_seen: int,
    next_data_point_checkpoint: int,
    stage_tag: str,
    epoch_idx: int,
    stage_epochs: int,
    scheduler=None,
) -> tuple[int, int, int]:
    loader = get_fineweb_loader(args, tokenizer)
    return train_one_epoch(
        model,
        tokenizer,
        loader,
        optimizer,
        device,
        args,
        global_step,
        data_points_seen,
        next_data_point_checkpoint,
        dataset_tag="fineweb",
        stage_tag=stage_tag,
        epoch_idx=epoch_idx,
        stage_epochs=stage_epochs,
        scheduler=scheduler,
    )


def evaluate(model, val_loader, device, args) -> float:
    """Evaluate model on validation dataset. Returns average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device, non_blocking=(device.type == "cuda"))
            labels = batch["labels"].to(device, non_blocking=(device.type == "cuda"))

            loss = compute_batch_loss(model, input_ids, labels)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    model.train()
    return avg_loss


def save_stage_checkpoint(
    model,
    tokenizer,
    args,
    optimizer,
    scheduler,
    scheduler_config: dict[str, Any] | None,
    stage_tag: str,
    epoch_idx: int,
    dataset_name: str,
    data_points_seen: int,
    global_step: int,
) -> None:
    ckpt_dir = Path(args.output_dir) / f"{args.model_type}-{stage_tag}-epoch-{epoch_idx}-{dataset_name}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _verify_mc_checkpoint_keys(model)
    torch.save(model.state_dict(), ckpt_dir / "pytorch_model.bin")
    tokenizer.save_pretrained(ckpt_dir)
    write_checkpoint_meta(
        ckpt_dir / "meta.txt",
        _training_checkpoint_meta(
            args,
            stage_tag=stage_tag,
            epoch_idx=epoch_idx,
            data_points=data_points_seen,
            global_step=global_step,
        ),
    )
    _save_training_bundle(
        ckpt_dir,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_config=scheduler_config,
        stage_tag=stage_tag,
        epoch_idx=epoch_idx,
        dataset_tag=dataset_name,
        batch_step=None,
        global_step=global_step,
        data_points_seen=data_points_seen,
        next_data_point_checkpoint=None,
    )
    print(f"Saved checkpoint to {ckpt_dir}")


def run_stage(
    model,
    tokenizer,
    optimizer,
    device,
    args,
    selected_datasets: list[str],
    stage_tag: str,
    stage_epochs: int,
    start_epoch_idx: int,
    global_step: int,
    data_points_seen: int,
    next_data_point_checkpoint: int,
    resume_training_bundle=None,
    wandb_run=None,
) -> tuple[int, int, int]:
    if start_epoch_idx > stage_epochs:
        print(
            f"Skipping stage={stage_tag}: resume start epoch {start_epoch_idx} exceeds configured "
            f"stage_epochs={stage_epochs}."
        )
        return global_step, data_points_seen, next_data_point_checkpoint

    scheduler = None
    scheduler_config: dict[str, Any] | None = None
    remaining_stage_epochs = (stage_epochs - start_epoch_idx) + 1
    total_updates = _estimate_stage_total_updates(args, tokenizer, selected_datasets, remaining_stage_epochs)
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler_name = args.lr_scheduler
    if _resume_bundle_matches_stage(resume_training_bundle, stage_tag):
        saved_scheduler_config = resume_training_bundle.get("scheduler_config") or {}
        if saved_scheduler_config:
            total_updates = int(saved_scheduler_config.get("total_updates", total_updates))
            warmup_steps = int(saved_scheduler_config.get("warmup_steps", warmup_steps))
            scheduler_name = str(saved_scheduler_config.get("scheduler_name", scheduler_name))
            print(
                f"Restoring scheduler geometry for stage={stage_tag}: "
                f"type={scheduler_name} total_updates={total_updates} warmup_steps={warmup_steps}"
            )
    scheduler_config = {
        "stage_tag": stage_tag,
        "scheduler_name": scheduler_name,
        "total_updates": total_updates,
        "warmup_steps": warmup_steps,
    }
    if total_updates > 0:
        scheduler = build_warmup_decay_scheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_updates,
            scheduler_name=scheduler_name,
        )
        if _resume_bundle_matches_stage(resume_training_bundle, stage_tag):
            scheduler_state_dict = resume_training_bundle.get("scheduler_state_dict")
            if scheduler_state_dict:
                scheduler.load_state_dict(scheduler_state_dict)
                print(f"Restored scheduler state for stage={stage_tag}.")
        print(
            f"Scheduler configured: type={scheduler_name} total_updates={total_updates} "
            f"warmup_steps={warmup_steps}"
        )
    for epoch_idx in range(start_epoch_idx, stage_epochs + 1):
        print(f"=== {stage_tag} epoch {epoch_idx}/{stage_epochs} ===")
        if args.dataset_strategy == "mix":
            mixed_loader = get_mixed_loader(args, tokenizer, selected_datasets)
            val_loaders = _build_stage_val_loaders(args, tokenizer, selected_datasets)
            if args.run_initial_validation:
                for val_name, val_loader in val_loaders:
                    val_loss = evaluate(model, val_loader, device, args)
                    val_ppl = perplexity_from_loss(val_loss)
                    print(
                        f"Validation | dataset={val_name} data_points={data_points_seen} "
                        f"loss={val_loss:.4f} ppl={val_ppl:.3e}"
                    )
                    val_key = re.sub(r"[^A-Za-z0-9_.-]", "_", str(val_name))
                    wandb_log(
                        wandb_run,
                        {
                            f"val/{val_key}/loss": val_loss,
                            f"val/{val_key}/ppl": val_ppl,
                            "val/data_points": data_points_seen,
                            "val/stage": stage_tag,
                            "val/epoch": epoch_idx,
                            "val/is_initial": 1,
                        },
                        step=global_step,
                    )
            global_step, data_points_seen, next_data_point_checkpoint = train_one_epoch(
                model,
                tokenizer,
                mixed_loader,
                optimizer,
                device,
                args,
                global_step,
                data_points_seen,
                next_data_point_checkpoint,
                dataset_tag="mixed",
                stage_tag=stage_tag,
                epoch_idx=epoch_idx,
                stage_epochs=stage_epochs,
                scheduler=scheduler,
                scheduler_config=scheduler_config,
                val_loaders=val_loaders,
                wandb_run=wandb_run,
            )
            save_stage_checkpoint(
                model,
                tokenizer,
                args,
                optimizer,
                scheduler,
                scheduler_config,
                stage_tag,
                epoch_idx,
                "mixed",
                data_points_seen,
                global_step,
            )
        else:
            for dataset_name in selected_datasets:
                if dataset_name == "wikitext":
                    loader = get_wikitext_loader(args, tokenizer)
                    dataset_tag = "wikitext"
                elif dataset_name == "fineweb":
                    loader = get_fineweb_loader(args, tokenizer)
                    dataset_tag = "fineweb"
                else:
                    continue

                val_loader = _build_val_loader_for_dataset(args, tokenizer, dataset_name)
                if args.run_initial_validation:
                    val_loss = evaluate(model, val_loader, device, args)
                    val_ppl = perplexity_from_loss(val_loss)
                    print(
                        f"Validation | dataset={dataset_name} data_points={data_points_seen} "
                        f"loss={val_loss:.4f} ppl={val_ppl:.3e}"
                    )
                    val_key = re.sub(r"[^A-Za-z0-9_.-]", "_", str(dataset_name))
                    wandb_log(
                        wandb_run,
                        {
                            f"val/{val_key}/loss": val_loss,
                            f"val/{val_key}/ppl": val_ppl,
                            "val/data_points": data_points_seen,
                            "val/stage": stage_tag,
                            "val/epoch": epoch_idx,
                            "val/is_initial": 1,
                        },
                        step=global_step,
                    )

                global_step, data_points_seen, next_data_point_checkpoint = train_one_epoch(
                    model=model,
                    tokenizer=tokenizer,
                    train_loader=loader,
                    optimizer=optimizer,
                    device=device,
                    args=args,
                    global_step=global_step,
                    data_points_seen=data_points_seen,
                    next_data_point_checkpoint=next_data_point_checkpoint,
                    dataset_tag=dataset_tag,
                    stage_tag=stage_tag,
                    epoch_idx=epoch_idx,
                    stage_epochs=stage_epochs,
                    scheduler=scheduler,
                    scheduler_config=scheduler_config,
                    val_loader=val_loader,
                    val_dataset_tag=dataset_name,
                    wandb_run=wandb_run,
                )
                save_stage_checkpoint(
                    model,
                    tokenizer,
                    args,
                    optimizer,
                    scheduler,
                    scheduler_config,
                    stage_tag,
                    epoch_idx,
                    dataset_name,
                    data_points_seen,
                    global_step,
                )
    return global_step, data_points_seen, next_data_point_checkpoint


def _build_val_loader_for_dataset(args, tokenizer, dataset_name: str) -> DataLoader:
    if dataset_name == "wikitext":
        return get_wikitext_val_loader(args, tokenizer)
    if dataset_name == "fineweb":
        return get_fineweb_val_loader(args, tokenizer)
    raise ValueError(f"Unsupported dataset for validation: {dataset_name}")


def _build_stage_val_loaders(args, tokenizer, selected_datasets: list[str]) -> list[tuple[str, DataLoader]]:
    loaders = []
    for dataset_name in selected_datasets:
        loaders.append((dataset_name, _build_val_loader_for_dataset(args, tokenizer, dataset_name)))
    return loaders


def _estimate_stage_total_updates(args, tokenizer, selected_datasets: list[str], stage_epochs: int) -> int:
    if stage_epochs <= 0:
        return 0
    if args.dataset_strategy == "mix":
        loader = get_mixed_loader(args, tokenizer, selected_datasets)
        updates_per_epoch = max(1, math.ceil(len(loader) / args.grad_accum_steps))
        return updates_per_epoch * stage_epochs

    total_updates = 0
    for dataset_name in selected_datasets:
        if dataset_name == "wikitext":
            loader = get_wikitext_loader(args, tokenizer)
        elif dataset_name == "fineweb":
            loader = get_fineweb_loader(args, tokenizer)
        else:
            continue
        total_updates += max(1, math.ceil(len(loader) / args.grad_accum_steps))
    return total_updates * stage_epochs


def main() -> None:
    args = parse_args()
    wandb_run = None
    resume_ckpt_meta: dict[str, str] = {}
    resume_training_bundle = None

    if args.resume_from_checkpoint:
        resume_ckpt_dir = Path(args.resume_from_checkpoint)
        if not resume_ckpt_dir.exists():
            raise FileNotFoundError(f"--resume-from-checkpoint does not exist: {resume_ckpt_dir}")
        resume_ckpt_meta = _read_checkpoint_meta(resume_ckpt_dir)
        resume_training_bundle = _load_training_bundle(resume_ckpt_dir)
        saved_geometry = parse_saved_training_geometry(resume_ckpt_meta)
        print_training_geometry(saved_geometry)
        align_args_with_checkpoint_geometry(
            args,
            resume_ckpt_meta,
            args._explicit_dests,
            TRAINING_ALIGNMENT_FIELDS,
            print_fn=print,
            context="training",
        )

    if args.block_size % 64 != 0:
        raise ValueError("--block-size should be a multiple of 64 to match Mamba2 chunking.")
    if args.mc_segment_size <= 0:
        raise ValueError("--mc-segment-size must be > 0.")
    if args.mc_segment_size > args.block_size:
        raise ValueError("--mc-segment-size should be <= --block-size.")
    if args.model_type in {"Mamba2MC", "Mamba2MCSelect"} and args.freeze_epochs > 0 and args.mc_segment_size >= args.block_size:
        raise ValueError(
            "For Mamba2MC freeze stage, --mc-segment-size must be < --block-size so MC params influence logits."
        )
    if args.mc_max_cached_segments < 0:
        raise ValueError("--mc-max-cached-segments must be >= 0.")
    if args.dataloader_num_workers < 0:
        raise ValueError("--dataloader-num-workers must be >= 0.")
    if args.dataloader_prefetch_factor <= 0:
        raise ValueError("--dataloader-prefetch-factor must be > 0.")
    if args.log_every_data_points <= 0:
        raise ValueError("--log-every-data-points must be > 0.")
    if args.wikitext_target_tokens < 0:
        raise ValueError("--wikitext-target-tokens must be >= 0.")
    if args.fineweb_target_tokens <= 0:
        raise ValueError("--fineweb-target-tokens must be > 0.")
    if not (0.0 < args.val_fraction_of_train <= 1.0):
        raise ValueError("--val-fraction-of-train must be in (0, 1].")
    if args.model_type in {"Mamba2MC", "Mamba2MCSelect"} and not (0.0 <= args.mc_online_bias_init <= 2.0):
        print(
            f"Warning: --mc-online-bias-init={args.mc_online_bias_init} is outside the recommended 0-2 range."
        )
    if args.mc_select_keep_top_k < 0:
        raise ValueError("--mc-select-keep-top-k must be >= 0.")
    _validate_training_geometry(args)
    _print_effective_training_geometry(args)

    torch.manual_seed(args.seed)
    device = get_device()
    _restore_rng_state_from_bundle(resume_training_bundle)

    if device.type == "cuda":
        # Throughput-oriented CUDA defaults.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Using device: {device}")
    print(
        "Tip: if train PPL stalls above your target, increase token budget/context "
        "(--fineweb-target-tokens, --wikitext-target-tokens, --block-size)."
    )

    print("Loading model and tokenizer...")
    if args.model_type == 'Mamba2':
        print("Loading Mamba2")
        model = load_mamba2_model(args.model_id, device=device, cache_dir=args.cache_dir)
    elif args.model_type == 'Mamba2MC':
        print("Loading Ours")
        model = load_our_model(
            args.model_id,
            device=device,
            cache_dir=args.cache_dir,
            segment_size=args.mc_segment_size,
            max_cached_segments=args.mc_max_cached_segments,
            detach_cached_segments=(not args.mc_backprop_history),
        )
        if hasattr(model, "online_bias"):
            with torch.no_grad():
                model.online_bias.fill_(args.mc_online_bias_init)
            print(f"Initialized online_bias to {args.mc_online_bias_init:.4f}")
    elif args.model_type == 'Mamba2MCSelect':
        print("Loading Ours (Selective Cache)")
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
        if hasattr(model, "online_bias"):
            with torch.no_grad():
                model.online_bias.fill_(args.mc_online_bias_init)
            print(f"Initialized online_bias to {args.mc_online_bias_init:.4f}")
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    _verify_mc_checkpoint_keys(model)
    _print_signal_param_trainability(model)

    if args.resume_from_checkpoint:
        ckpt_dir = Path(args.resume_from_checkpoint)
        weights_path = ckpt_dir / "pytorch_model.bin"
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"--resume-from-checkpoint does not exist: {ckpt_dir}")
        if not weights_path.exists():
            raise FileNotFoundError(f"Checkpoint weights not found: {weights_path}")
        print(f"Loading checkpoint weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Checkpoint weights loaded.")

    tokenizer_source = args.tokenizer_id
    if args.resume_from_checkpoint:
        ckpt_dir = Path(args.resume_from_checkpoint)
        if (ckpt_dir / "tokenizer.json").exists() or (ckpt_dir / "tokenizer_config.json").exists():
            tokenizer_source = str(ckpt_dir)
            print(f"Loading tokenizer from checkpoint dir: {ckpt_dir}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, cache_dir=args.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token

    selected_datasets = normalize_dataset_choice(args.train_datasets)
    print(f"Selected datasets: {selected_datasets}")
    wandb_run = init_wandb_run(args) if wandb_enabled_from_env() else None

    global_step = 0
    data_points_seen = 0
    next_data_point_checkpoint = args.data_point_checkpoint_interval
    optimizer = None
    resume_stage = None
    resume_epoch = 0
    if args.resume_from_checkpoint:
        resume_ckpt_dir = Path(args.resume_from_checkpoint)
        ckpt_meta = resume_ckpt_meta or _read_checkpoint_meta(resume_ckpt_dir)
        resume_stage, resume_epoch = _resolve_resume_stage_epoch(resume_ckpt_dir, ckpt_meta)
        if "global_step" in ckpt_meta:
            try:
                global_step = int(ckpt_meta["global_step"])
            except ValueError:
                print(f"Warning: invalid global_step in checkpoint meta: {ckpt_meta['global_step']}")
        if "data_points" in ckpt_meta:
            try:
                data_points_seen = int(ckpt_meta["data_points"])
            except ValueError:
                print(f"Warning: invalid data_points in checkpoint meta: {ckpt_meta['data_points']}")
        if args.data_point_checkpoint_interval > 0:
            next_data_point_checkpoint = (
                (data_points_seen // args.data_point_checkpoint_interval) + 1
            ) * args.data_point_checkpoint_interval
        if resume_training_bundle and resume_training_bundle.get("next_data_point_checkpoint") is not None:
            next_data_point_checkpoint = int(resume_training_bundle["next_data_point_checkpoint"])
        print(
            f"Resuming counters from checkpoint: global_step={global_step}, "
            f"data_points_seen={data_points_seen}, next_checkpoint={next_data_point_checkpoint}"
        )
        if resume_stage is not None:
            print(f"Resume stage info: stage={resume_stage}, epoch={resume_epoch}")
        else:
            print("Resume stage info: not found in checkpoint metadata or directory name.")

    freeze_start_epoch = 1
    full_start_epoch = 1
    skip_freeze_stage = False
    if args.resume_from_checkpoint and resume_stage in {"freeze", "full"}:
        if resume_stage == "freeze":
            if _checkpoint_is_data_point_ckpt(Path(args.resume_from_checkpoint)):
                freeze_start_epoch = max(1, resume_epoch)
            else:
                freeze_start_epoch = max(1, resume_epoch + 1)
        elif resume_stage == "full":
            skip_freeze_stage = True
            if _checkpoint_is_data_point_ckpt(Path(args.resume_from_checkpoint)):
                full_start_epoch = max(1, resume_epoch)
            else:
                full_start_epoch = max(1, resume_epoch + 1)

    if args.freeze_epochs > 0 and not skip_freeze_stage:
        policy_model_name, trainable_names, frozen_names = configure_stage1_trainability_by_model_name(model, args)
        print(f"Freeze stage model policy applied: {policy_model_name}")
        print(f"Freeze stage trainable params count: {len(trainable_names)}")
        print(f"Freeze stage frozen params count: {len(frozen_names)}")
        _print_signal_param_trainability(model)
        if args.model_type in {"Mamba2MC", "Mamba2MCSelect"} and len(trainable_names) <= 2:
            raise ValueError(
                "Mamba2MC freeze stage has <=2 trainable params under current --mc-freeze-train-mode. "
                "Use --mc-freeze-train-mode mc_plus_norm (recommended), all, or set --freeze-epochs 0."
            )
        optimizer = build_optimizer(model, args)
        if _resume_bundle_matches_stage(resume_training_bundle, "freeze"):
            optimizer_state_dict = resume_training_bundle.get("optimizer_state_dict")
            if optimizer_state_dict:
                optimizer.load_state_dict(optimizer_state_dict)
                _move_optimizer_state_to_device(optimizer, device)
                print("Restored optimizer state for stage=freeze.")
        global_step, data_points_seen, next_data_point_checkpoint = run_stage(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            device=device,
            args=args,
            selected_datasets=selected_datasets,
            stage_tag="freeze",
            stage_epochs=args.freeze_epochs,
            start_epoch_idx=freeze_start_epoch,
            global_step=global_step,
            data_points_seen=data_points_seen,
            next_data_point_checkpoint=next_data_point_checkpoint,
            resume_training_bundle=resume_training_bundle,
            wandb_run=wandb_run,
        )
    elif skip_freeze_stage:
        print("Skipping freeze stage because resume checkpoint is from full stage.")

    if args.full_finetune_epochs > 0:
        unfreeze_all_params(model)
        print("Unfroze all model parameters for full fine-tuning stage.")
        _print_signal_param_trainability(model)
        optimizer = build_optimizer(model, args)
        if _resume_bundle_matches_stage(resume_training_bundle, "full"):
            optimizer_state_dict = resume_training_bundle.get("optimizer_state_dict")
            if optimizer_state_dict:
                optimizer.load_state_dict(optimizer_state_dict)
                _move_optimizer_state_to_device(optimizer, device)
                print("Restored optimizer state for stage=full.")
        global_step, data_points_seen, next_data_point_checkpoint = run_stage(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            device=device,
            args=args,
            selected_datasets=selected_datasets,
            stage_tag="full",
            stage_epochs=args.full_finetune_epochs,
            start_epoch_idx=full_start_epoch,
            global_step=global_step,
            data_points_seen=data_points_seen,
            next_data_point_checkpoint=next_data_point_checkpoint,
            resume_training_bundle=resume_training_bundle,
            wandb_run=wandb_run,
        )

    final_dir = Path(args.output_dir) / f"{args.model_type}-final"
    final_dir.mkdir(parents=True, exist_ok=True)
    _verify_mc_checkpoint_keys(model)
    torch.save(model.state_dict(), final_dir / "pytorch_model.bin")
    tokenizer.save_pretrained(final_dir)
    write_checkpoint_meta(
        final_dir / "meta.txt",
        _training_checkpoint_meta(
            args,
            stage_tag="final",
            data_points=data_points_seen,
            global_step=global_step,
        ),
    )
    _save_training_bundle(
        final_dir,
        optimizer=optimizer,
        scheduler=None,
        scheduler_config=None,
        stage_tag="final",
        epoch_idx=None,
        dataset_tag=None,
        batch_step=None,
        global_step=global_step,
        data_points_seen=data_points_seen,
        next_data_point_checkpoint=next_data_point_checkpoint,
    )
    print(f"Training complete. Final model saved to {final_dir}")
    wandb_log(
        wandb_run,
        {
            "final/global_step": global_step,
            "final/data_points": data_points_seen,
            "final/checkpoint_dir": str(final_dir),
        },
        step=global_step,
    )
    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception as exc:
            print(f"Warning: W&B finish failed ({exc}).")


if __name__ == "__main__":
    main()
