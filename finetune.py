import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset, concatenate_datasets, load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
    parser.add_argument("--output-dir", type=str, default="./checkpoints/mamba2-finetune")
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
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--block-size", type=int, default=128, help="Should be a multiple of 64 for this model.")
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
        "--mc-train-mode",
        type=str,
        choices=["throughput", "memory"],
        default="throughput",
        help="throughput: one backward per batch (faster, more memory). memory: token-streaming backward (slower, lower memory).",
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
    parser.add_argument("--log-every", type=int, default=20)

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
        "--fineweb-config",
        type=str,
        default="CC-MAIN-2024-10",
        help="FineWeb subset/config to stream.",
    )
    parser.add_argument(
        "--fineweb-target-mb",
        type=int,
        default=100,
        help="Target raw UTF-8 text size in MB (script avoids going over this cap).",
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
    return parser.parse_args()


def tokenize_and_pack_text_dataset(text_dataset: Dataset, tokenizer, block_size: int) -> Dataset:
    def tokenize_fn(examples):
        texts = [t for t in examples["text"] if t and not t.isspace()]
        return tokenizer(texts, return_attention_mask=False)

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

    tokenized = text_dataset.map(tokenize_fn, batched=True, remove_columns=text_dataset.column_names)
    packed = tokenized.map(group_texts, batched=True)
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


def get_wikitext_loader(args, tokenizer) -> DataLoader:
    dataset = load_dataset("Salesforce/wikitext", args.wikitext_config, split="train", cache_dir=args.cache_dir)

    if args.max_train_samples_wikitext > 0:
        dataset = dataset.select(range(min(args.max_train_samples_wikitext, len(dataset))))

    packed = tokenize_and_pack_text_dataset(dataset, tokenizer, args.block_size)
    return build_dataloader(packed, args, shuffle=True)


def get_wikitext_text_dataset(args) -> Dataset:
    dataset = load_dataset("Salesforce/wikitext", args.wikitext_config, split="train", cache_dir=args.cache_dir)
    if args.max_train_samples_wikitext > 0:
        dataset = dataset.select(range(min(args.max_train_samples_wikitext, len(dataset))))
    return dataset


def get_fineweb_loader(args, tokenizer) -> DataLoader:
    target_bytes = args.fineweb_target_mb * 1024 * 1024
    streamed = load_dataset(
        "HuggingFaceFW/fineweb",
        name=args.fineweb_config,
        split="train",
        streaming=True,
        cache_dir=args.cache_dir,
    )

    texts = []
    total_bytes = 0

    for i, sample in enumerate(streamed):
        text = sample.get("text", "")
        if not text or text.isspace():
            continue

        text_bytes = len(text.encode("utf-8"))
        if total_bytes + text_bytes > target_bytes:
            break

        texts.append(text)
        total_bytes += text_bytes

        if args.fineweb_max_stream_samples > 0 and (i + 1) >= args.fineweb_max_stream_samples:
            break

    if not texts:
        raise RuntimeError("No FineWeb text was collected. Try another --fineweb-config.")

    print(
        f"Collected FineWeb slice: docs={len(texts)}, bytes={total_bytes} "
        f"(~{total_bytes / (1024 * 1024):.2f} MB, cap={args.fineweb_target_mb} MB)"
    )

    text_dataset = Dataset.from_dict({"text": texts})
    packed = tokenize_and_pack_text_dataset(text_dataset, tokenizer, args.block_size)
    return build_dataloader(packed, args, shuffle=True)


def get_fineweb_text_dataset(args) -> Dataset:
    target_bytes = args.fineweb_target_mb * 1024 * 1024
    streamed = load_dataset(
        "HuggingFaceFW/fineweb",
        name=args.fineweb_config,
        split="train",
        streaming=True,
        cache_dir=args.cache_dir,
    )

    texts = []
    total_bytes = 0

    for i, sample in enumerate(streamed):
        text = sample.get("text", "")
        if not text or text.isspace():
            continue

        text_bytes = len(text.encode("utf-8"))
        if total_bytes + text_bytes > target_bytes:
            break

        texts.append(text)
        total_bytes += text_bytes

        if args.fineweb_max_stream_samples > 0 and (i + 1) >= args.fineweb_max_stream_samples:
            break

    if not texts:
        raise RuntimeError("No FineWeb text was collected. Try another --fineweb-config.")

    print(
        f"Collected FineWeb slice: docs={len(texts)}, bytes={total_bytes} "
        f"(~{total_bytes / (1024 * 1024):.2f} MB, cap={args.fineweb_target_mb} MB)"
    )
    return Dataset.from_dict({"text": texts})


def get_wikitext_val_loader(args, tokenizer) -> DataLoader:
    dataset = load_dataset("Salesforce/wikitext", args.wikitext_config, split="validation", cache_dir=args.cache_dir)
    packed = tokenize_and_pack_text_dataset(dataset, tokenizer, args.block_size)
    return build_dataloader(packed, args, shuffle=False)


def get_fineweb_val_loader(args, tokenizer) -> DataLoader:
    # For fineweb validation, use a small fixed subset
    target_bytes = 50 * 1024 * 1024  # 50MB for validation
    streamed = load_dataset(
        "HuggingFaceFW/fineweb",
        name=args.fineweb_config,
        split="train",
        streaming=True,
        cache_dir=args.cache_dir,
    )

    texts = []
    total_bytes = 0
    for i, sample in enumerate(streamed):
        text = sample.get("text", "")
        if not text or text.isspace():
            continue
        text_bytes = len(text.encode("utf-8"))
        if total_bytes + text_bytes > target_bytes:
            break
        texts.append(text)
        total_bytes += text_bytes

    if not texts:
        raise RuntimeError("No FineWeb validation text was collected.")

    text_dataset = Dataset.from_dict({"text": texts})
    packed = tokenize_and_pack_text_dataset(text_dataset, tokenizer, args.block_size)
    return build_dataloader(packed, args, shuffle=False)


def get_mixed_loader(args, tokenizer, selected_datasets: list[str]) -> DataLoader:
    text_datasets = []

    if "wikitext" in selected_datasets:
        text_datasets.append(get_wikitext_text_dataset(args))
    if "fineweb" in selected_datasets:
        text_datasets.append(get_fineweb_text_dataset(args))

    if not text_datasets:
        raise ValueError("No datasets selected for mixed training.")

    if len(text_datasets) == 1:
        mixed_text_dataset = text_datasets[0]
    else:
        mixed_text_dataset = concatenate_datasets(text_datasets)

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
    """Return True for MC-only params intended to train during freeze stage."""
    return (
        name == "W"
        or name.endswith(".W")
        or "W_u" in name
        or "online_bias" in name
    )


def configure_stage1_trainability_by_model_name(model) -> tuple[str, list[str], list[str]]:
    model_name = model.__class__.__name__

    if model_name == "Mamba2LMHeadModel":
        # Current request: for the current Mamba2 model, keep all params trainable.
        for _, param in model.named_parameters():
            param.requires_grad = True
    elif model_name == "Mamba2MCLMHeadModel":
        # Freeze original model parameters, train only MC-specific params.
        # Supports both legacy MC param names and current state-only MC `W`.
        for name, param in model.named_parameters():
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

def unfreeze_all_params(model) -> None:
    for _, param in model.named_parameters():
        param.requires_grad = True


def build_optimizer(model, args) -> AdamW:
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found for optimizer.")
    return AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)


def build_linear_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    try:
        from transformers import get_linear_schedule_with_warmup
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    except ImportError:
        from transformers.optimization import WarmupLinearSchedule
        return WarmupLinearSchedule(
            optimizer,
            warmup_steps=warmup_steps,
            t_total=total_steps,
        )


def save_data_point_checkpoint(
    model,
    tokenizer,
    checkpoint_dir: str,
    data_points_mark: int,
    stage_tag: str,
    epoch_idx: int,
) -> None:
    ckpt_dir = Path(checkpoint_dir) / f"data-points-{data_points_mark}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "pytorch_model.bin")
    tokenizer.save_pretrained(ckpt_dir)
    meta_path = ckpt_dir / "meta.txt"
    meta_path.write_text(
        f"stage={stage_tag}\nepoch={epoch_idx}\ndata_points={data_points_mark}\n",
        encoding="utf-8",
    )
    print(f"Saved data-point checkpoint to {ckpt_dir}")


def _is_mc_model(model) -> bool:
    return model.__class__.__name__ == "Mamba2MCLMHeadModel"


def _is_mc_only_trainable(model) -> bool:
    if not _is_mc_model(model):
        return False
    for name, param in model.named_parameters():
        if param.requires_grad and not _is_mc_specific_trainable_param(name):
            return False
    return True


def compute_batch_loss(model, input_ids, labels) -> torch.Tensor:
    """Compute autoregressive CE loss; uses streamed-token path for Mamba2MC to save memory."""
    if not _is_mc_model(model):
        logits, _ = model(input_ids)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    batch_size, seqlen = input_ids.shape
    if seqlen < 2:
        raise ValueError("Input sequence length must be at least 2 for autoregressive loss.")

    caches = model.alloc_cache(batch_size=batch_size)
    total_nll = torch.zeros((), device=input_ids.device)
    total_valid_tokens = (labels[:, 1:] != -100).sum()
    if int(total_valid_tokens.item()) == 0:
        raise RuntimeError("No valid tokens available to compute loss.")

    for pos in range(seqlen - 1):
        step_logits, caches = model.step(input_ids[:, pos : pos + 1], caches)
        step_targets = labels[:, pos + 1]

        step_nll = F.cross_entropy(
            step_logits.squeeze(1),
            step_targets,
            ignore_index=-100,
            reduction="sum",
        )

        total_nll = total_nll + step_nll

    return total_nll / total_valid_tokens.to(total_nll.dtype)


def backward_batch_loss_mc_streaming(
    model,
    input_ids,
    labels,
    grad_accum_steps: int,
) -> torch.Tensor:
    """Memory-efficient MC freeze-stage loss/backward: backprop each token immediately."""
    batch_size, seqlen = input_ids.shape
    if seqlen < 2:
        raise ValueError("Input sequence length must be at least 2 for autoregressive loss.")

    caches = model.alloc_cache(batch_size=batch_size)
    total_valid_tokens = (labels[:, 1:] != -100).sum()
    if int(total_valid_tokens.item()) == 0:
        raise RuntimeError("No valid tokens available to compute loss.")

    total_loss_value = torch.zeros((), device=input_ids.device)
    any_grad_step = False

    for pos in range(seqlen - 1):
        step_logits, caches = model.step(input_ids[:, pos : pos + 1], caches)
        step_targets = labels[:, pos + 1]

        step_nll = F.cross_entropy(
            step_logits.squeeze(1),
            step_targets,
            ignore_index=-100,
            reduction="sum",
        )

        step_loss = step_nll / total_valid_tokens.to(step_nll.dtype)
        total_loss_value = total_loss_value + step_loss.detach()

        # In MC-only freeze stage, early tokens can legitimately have no grad path
        # (before the first segment is cached). Backprop only when a graph exists.
        if step_loss.requires_grad:
            (step_loss / grad_accum_steps).backward()
            any_grad_step = True

    if not any_grad_step:
        raise RuntimeError(
            "Loss has no grad_fn for all tokens in this batch. In freeze stage, this usually means "
            "trainable MC params did not influence logits. Try setting --mc-segment-size smaller than "
            "--block-size (for example, --mc-segment-size 64 with --block-size 256), or unfreeze more params."
        )

    return total_loss_value


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
    val_loader=None,
) -> tuple[int, int, int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    num_batches = len(train_loader)
    updates_this_epoch = max(1, math.ceil(num_batches / args.grad_accum_steps))
    warmup_steps = int(updates_this_epoch * args.warmup_ratio)
    scheduler = build_linear_warmup_scheduler(optimizer, warmup_steps, updates_this_epoch)

    running_loss = torch.zeros((), device=device)
    next_val_checkpoint = 1000
    mc_only_memory_mode = _is_mc_only_trainable(model) and args.mc_train_mode == "memory"
    trainable_params = [p for p in model.parameters() if p.requires_grad]

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

        if mc_only_memory_mode:
            loss_value = backward_batch_loss_mc_streaming(
                model=model,
                input_ids=input_ids,
                labels=labels,
                grad_accum_steps=args.grad_accum_steps,
            )
            running_loss = running_loss + loss_value
        else:
            loss = compute_batch_loss(model, input_ids, labels)
            if not loss.requires_grad:
                raise RuntimeError(
                    "Loss has no grad_fn. In freeze stage, this usually means trainable MC params "
                    "did not influence logits. Try setting --mc-segment-size smaller than --block-size "
                    "(for example, --mc-segment-size 64 with --block-size 256), or unfreeze more params."
                )
            loss = loss / args.grad_accum_steps
            loss.backward()
            running_loss = running_loss + (loss.detach() * args.grad_accum_steps)

        should_update = (step % args.grad_accum_steps == 0) or (step == num_batches)
        if should_update:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

            if global_step % args.log_every == 0:
                avg_loss = float((running_loss / args.log_every).item())
                ppl = math.exp(min(avg_loss, 20))
                lr = scheduler.get_last_lr()[0]
                log_msg = (
                    f"stage={stage_tag} dataset={dataset_tag} step={global_step} "
                    f"loss={avg_loss:.4f} ppl={ppl:.2f} lr={lr:.2e}"
                )
                print(log_msg)
                progress.set_postfix_str(f"loss={avg_loss:.4f}")
                running_loss = torch.zeros((), device=device)

        if args.data_point_checkpoint_interval > 0:
            while data_points_seen >= next_data_point_checkpoint:
                save_data_point_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    checkpoint_dir=args.checkpoint_dir,
                    data_points_mark=next_data_point_checkpoint,
                    stage_tag=stage_tag,
                    epoch_idx=epoch_idx,
                )
                next_data_point_checkpoint += args.data_point_checkpoint_interval

        # Validation every 1k datapoints
        if val_loader is not None and data_points_seen >= next_val_checkpoint:
            val_loss = evaluate(model, val_loader, device, args)
            val_ppl = math.exp(min(val_loss, 20))
            print(
                f"Validation | data_points={data_points_seen} "
                f"loss={val_loss:.4f} ppl={val_ppl:.2f}"
            )
            next_val_checkpoint += 1000
            model.train()

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


def save_stage_checkpoint(model, tokenizer, args, stage_tag: str, epoch_idx: int, dataset_name: str) -> None:
    ckpt_dir = Path(args.output_dir) / f"{stage_tag}-epoch-{epoch_idx}-{dataset_name}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "pytorch_model.bin")
    tokenizer.save_pretrained(ckpt_dir)
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
    global_step: int,
    data_points_seen: int,
    next_data_point_checkpoint: int,
) -> tuple[int, int, int]:
    for epoch_idx in range(1, stage_epochs + 1):
        print(f"=== {stage_tag} epoch {epoch_idx}/{stage_epochs} ===")
        if args.dataset_strategy == "mix":
            mixed_loader = get_mixed_loader(args, tokenizer, selected_datasets)
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
            )
            save_stage_checkpoint(model, tokenizer, args, stage_tag, epoch_idx, "mixed")
        else:
            for dataset_name in selected_datasets:
                if dataset_name == "wikitext":
                    global_step, data_points_seen, next_data_point_checkpoint = finetune_one_epoch_on_wikitext(
                        model,
                        tokenizer,
                        optimizer,
                        device,
                        args,
                        global_step,
                        data_points_seen,
                        next_data_point_checkpoint,
                        stage_tag,
                        epoch_idx,
                        stage_epochs,
                    )
                elif dataset_name == "fineweb":
                    global_step, data_points_seen, next_data_point_checkpoint = finetune_one_epoch_on_fineweb(
                        model,
                        tokenizer,
                        optimizer,
                        device,
                        args,
                        global_step,
                        data_points_seen,
                        next_data_point_checkpoint,
                        stage_tag,
                        epoch_idx,
                        stage_epochs,
                    )
                save_stage_checkpoint(model, tokenizer, args, stage_tag, epoch_idx, dataset_name)
    return global_step, data_points_seen, next_data_point_checkpoint


def main() -> None:
    args = parse_args()

    if args.block_size % 64 != 0:
        raise ValueError("--block-size should be a multiple of 64 to match Mamba2 chunking.")
    if args.mc_segment_size <= 0:
        raise ValueError("--mc-segment-size must be > 0.")
    if args.mc_segment_size > args.block_size:
        raise ValueError("--mc-segment-size should be <= --block-size.")
    if args.mc_max_cached_segments < 0:
        raise ValueError("--mc-max-cached-segments must be >= 0.")
    if args.dataloader_num_workers < 0:
        raise ValueError("--dataloader-num-workers must be >= 0.")
    if args.dataloader_prefetch_factor <= 0:
        raise ValueError("--dataloader-prefetch-factor must be > 0.")

    torch.manual_seed(args.seed)
    device = get_device()

    if device.type == "cuda":
        # Throughput-oriented CUDA defaults.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Using device: {device}")

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
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id, cache_dir=args.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token

    selected_datasets = normalize_dataset_choice(args.train_datasets)
    print(f"Selected datasets: {selected_datasets}")

    global_step = 0
    data_points_seen = 0
    next_data_point_checkpoint = args.data_point_checkpoint_interval

    if args.freeze_epochs > 0:
        policy_model_name, trainable_names, frozen_names = configure_stage1_trainability_by_model_name(model)
        print(f"Freeze stage model policy applied: {policy_model_name}")
        print(f"Freeze stage trainable params count: {len(trainable_names)}")
        print(f"Freeze stage frozen params count: {len(frozen_names)}")
        optimizer = build_optimizer(model, args)
        global_step, data_points_seen, next_data_point_checkpoint = run_stage(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            device=device,
            args=args,
            selected_datasets=selected_datasets,
            stage_tag="freeze",
            stage_epochs=args.freeze_epochs,
            global_step=global_step,
            data_points_seen=data_points_seen,
            next_data_point_checkpoint=next_data_point_checkpoint,
        )

    if args.full_finetune_epochs > 0:
        unfreeze_all_params(model)
        print("Unfroze all model parameters for full fine-tuning stage.")
        optimizer = build_optimizer(model, args)
        global_step, data_points_seen, next_data_point_checkpoint = run_stage(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            device=device,
            args=args,
            selected_datasets=selected_datasets,
            stage_tag="full",
            stage_epochs=args.full_finetune_epochs,
            global_step=global_step,
            data_points_seen=data_points_seen,
            next_data_point_checkpoint=next_data_point_checkpoint,
        )

    final_dir = Path(args.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), final_dir / "pytorch_model.bin")
    tokenizer.save_pretrained(final_dir)
    print(f"Training complete. Final model saved to {final_dir}")


if __name__ == "__main__":
    main()
