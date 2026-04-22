import json
import math
from dataclasses import dataclass
from typing import Iterable, TypeAlias, cast

import torch
import torch.nn.functional as F
from torch import LongTensor, Tensor, nn

from mamba2 import InferenceCache, Mamba2, Mamba2Config, RMSNorm

Device: TypeAlias = str | torch.device | None


@dataclass
class MCInferenceCache:
    layer_caches: list[InferenceCache]
    segment_buffer: list[Tensor]  # each tensor is (batch, d_model)
    current_segment_sum: Tensor  # (batch, d_model)
    current_segment_len: int


class Mamba2MCLMHeadModel(nn.Module):
    def __init__(
        self,
        args: Mamba2Config,
        device: Device = None,
        segment_size: int = 64,
        max_cached_segments: int = 16,
        detach_cached_segments: bool = True,
    ):
        super().__init__()
        self.args = args
        self.device = device
        self.segment_size = segment_size
        self.max_cached_segments = max_cached_segments
        self.detach_cached_segments = detach_cached_segments

        self.backbone = nn.ModuleDict(
            dict(
                embedding=nn.Embedding(args.vocab_size, args.d_model, device=device),
                layers=nn.ModuleList(
                    [
                        nn.ModuleDict(
                            dict(
                                mixer=Mamba2(args, device=device),
                                norm=RMSNorm(args.d_model, device=device),
                            )
                        )
                        for _ in range(args.n_layer)
                    ]
                ),
                norm_f=RMSNorm(args.d_model, device=device),
            )
        )
        self.lm_head = nn.Linear(
            args.d_model, args.vocab_size, bias=False, device=device
        )
        self.lm_head.weight = self.backbone.embedding.weight

        # Trainable matrix used to compute weighting ratios over cached hidden states.
        self.W = nn.Parameter(torch.empty(args.d_model, args.d_model, device=device))
        nn.init.xavier_uniform_(self.W)
        # Scalar gate to blend current hidden state and cached weighted history.
        self.online_bias = nn.Parameter(torch.zeros((), device=device))

    def _forward_backbone_full(self, input_ids: LongTensor) -> tuple[Tensor, list[InferenceCache]]:
        """Fast path: run full-sequence backbone in parallel, mirroring Mamba2."""
        x = self.backbone.embedding(input_ids)
        layer_caches: list[InferenceCache] = []
        for layer in self.backbone.layers:
            y, layer_cache = layer.mixer(layer.norm(x), None)
            x = x + y
            layer_caches.append(layer_cache)
        x = self.backbone.norm_f(x)
        return x, layer_caches

    @staticmethod
    def from_pretrained(
        huggingface_model_id: str,
        device: Device = None,
        cache_dir: str = None,
        segment_size: int = 64,
        max_cached_segments: int = 16,
        detach_cached_segments: bool = True,
    ):
        try:
            from transformers.utils import CONFIG_NAME, WEIGHTS_NAME
            from transformers.utils.hub import cached_file
        except ImportError:
            from transformers.file_utils import CONFIG_NAME, WEIGHTS_NAME, cached_path

            def cached_file(model_id: str, filename: str, cache_dir: str = None):
                url = f"https://huggingface.co/{model_id}/resolve/main/{filename}"
                return cached_path(url, cache_dir=cache_dir)

        config_path = cached_file(huggingface_model_id, CONFIG_NAME, cache_dir=cache_dir)
        assert config_path, "Failed to get huggingface config file"

        state_dict_path = cached_file(huggingface_model_id, WEIGHTS_NAME, cache_dir=cache_dir)
        assert state_dict_path, "Failed to get huggingface state dict file"

        config = json.load(open(config_path))
        args = Mamba2Config(
            d_model=config["d_model"],
            n_layer=config["n_layer"],
            vocab_size=config["vocab_size"],
            pad_vocab_size_multiple=config["pad_vocab_size_multiple"],
        )

        map_location = "cpu" if device is None else device
        state_dict = torch.load(
            state_dict_path, weights_only=True, map_location=map_location, mmap=True
        )

        model = Mamba2MCLMHeadModel(
            args=args,
            device=device,
            segment_size=segment_size,
            max_cached_segments=max_cached_segments,
            detach_cached_segments=detach_cached_segments,
        )
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        allowed_missing = {"W", "online_bias"}
        unresolved_missing = set(missing_keys) - allowed_missing
        if unresolved_missing:
            raise RuntimeError(
                f"Unexpected missing keys while loading pretrained weights: {sorted(unresolved_missing)}"
            )
        if unexpected_keys:
            raise RuntimeError(
                f"Unexpected keys while loading pretrained weights: {sorted(unexpected_keys)}"
            )

        model.eval()
        return model

    def alloc_cache(self, batch_size: int) -> MCInferenceCache:
        device = self.backbone.embedding.weight.device
        layer_caches = [
            InferenceCache.alloc(batch_size, self.args, device=device)
            for _ in range(self.args.n_layer)
        ]
        current_segment_sum = torch.zeros(
            batch_size,
            self.args.d_model,
            device=device,
            dtype=self.backbone.embedding.weight.dtype,
        )
        return MCInferenceCache(
            layer_caches=layer_caches,
            segment_buffer=[],
            current_segment_sum=current_segment_sum,
            current_segment_len=0,
        )

    def _weighted_history_mix(self, hidden_t: Tensor, cache: MCInferenceCache) -> Tensor:
        if len(cache.segment_buffer) == 0:
            return hidden_t

        history = torch.stack(cache.segment_buffer, dim=1)  # (batch, n_seg, d_model)
        query = hidden_t @ self.W  # (batch, d_model)
        scores = torch.einsum("bd,bnd->bn", query, history) / math.sqrt(self.args.d_model)
        ratios = F.softmax(scores, dim=-1)
        weighted_hist = torch.einsum("bn,bnd->bd", ratios, history)

        gate = torch.sigmoid(self.online_bias)
        return gate * hidden_t + (1.0 - gate) * weighted_hist

    def _update_segment_cache(self, hidden_t: Tensor, cache: MCInferenceCache) -> None:
        cache.current_segment_sum = cache.current_segment_sum + hidden_t
        cache.current_segment_len += 1

        if cache.current_segment_len < self.segment_size:
            return

        segment_hidden = cache.current_segment_sum / float(cache.current_segment_len)
        if self.detach_cached_segments:
            segment_hidden = segment_hidden.detach()
        cache.segment_buffer.append(segment_hidden)

        if self.max_cached_segments > 0 and len(cache.segment_buffer) > self.max_cached_segments:
            cache.segment_buffer = cache.segment_buffer[-self.max_cached_segments :]

        cache.current_segment_sum = torch.zeros_like(cache.current_segment_sum)
        cache.current_segment_len = 0

    def step(
        self,
        input_ids: LongTensor,
        cache: MCInferenceCache,
    ) -> tuple[Tensor, MCInferenceCache]:
        if input_ids.ndim != 2 or input_ids.shape[1] != 1:
            raise ValueError("step expects shape (batch, 1) input_ids")

        x = self.backbone.embedding(input_ids)
        for i, layer in enumerate(self.backbone.layers):
            y, cache.layer_caches[i] = layer.mixer(layer.norm(x), cache.layer_caches[i])
            x = x + y

        x = self.backbone.norm_f(x)
        hidden_t = x[:, 0, :]  # (batch, d_model)

        mixed_hidden = self._weighted_history_mix(hidden_t, cache)
        logits = self.lm_head(mixed_hidden).unsqueeze(1)

        self._update_segment_cache(hidden_t, cache)
        return logits, cache

    def forward(
        self,
        input_ids: LongTensor,
        cache: MCInferenceCache | None = None,
    ) -> tuple[Tensor, MCInferenceCache]:
        batch_size, seqlen = input_ids.shape
        # Fast training/eval path: no external cache and chunk-aligned sequence.
        # This preserves Mamba2's parallel backbone execution speed.
        if cache is None and seqlen % self.args.chunk_size == 0:
            hidden, layer_caches = self._forward_backbone_full(input_ids)  # (b, l, d)

            segment_buffer: list[Tensor] = []
            current_segment_sum = torch.zeros(
                batch_size,
                self.args.d_model,
                device=hidden.device,
                dtype=hidden.dtype,
            )
            current_segment_len = 0
            gate = torch.sigmoid(self.online_bias)
            logits_chunks = []

            for start in range(0, seqlen, self.segment_size):
                end = min(seqlen, start + self.segment_size)
                hidden_chunk = hidden[:, start:end, :]  # (b, l_seg, d)

                if len(segment_buffer) == 0:
                    mixed_chunk = hidden_chunk
                else:
                    history = torch.stack(segment_buffer, dim=1)  # (b, n_seg, d)
                    query = torch.matmul(hidden_chunk, self.W)  # (b, l_seg, d)
                    scores = (
                        torch.einsum("bld,bnd->bln", query, history)
                        / math.sqrt(self.args.d_model)
                    )
                    ratios = F.softmax(scores, dim=-1)
                    weighted_hist = torch.einsum("bln,bnd->bld", ratios, history)
                    mixed_chunk = gate * hidden_chunk + (1.0 - gate) * weighted_hist

                logits_chunks.append(self.lm_head(mixed_chunk))

                # Update segment cache state for compatibility with returned cache semantics.
                for pos in range(hidden_chunk.shape[1]):
                    hidden_t = hidden_chunk[:, pos, :]
                    current_segment_sum = current_segment_sum + hidden_t
                    current_segment_len += 1
                    if current_segment_len >= self.segment_size:
                        segment_hidden = current_segment_sum / float(current_segment_len)
                        if self.detach_cached_segments:
                            segment_hidden = segment_hidden.detach()
                        segment_buffer.append(segment_hidden)
                        if (
                            self.max_cached_segments > 0
                            and len(segment_buffer) > self.max_cached_segments
                        ):
                            segment_buffer = segment_buffer[-self.max_cached_segments :]
                        current_segment_sum = torch.zeros_like(current_segment_sum)
                        current_segment_len = 0

            logits = torch.cat(logits_chunks, dim=1)
            final_cache = MCInferenceCache(
                layer_caches=layer_caches,
                segment_buffer=segment_buffer,
                current_segment_sum=current_segment_sum,
                current_segment_len=current_segment_len,
            )
            return logits, final_cache

        # Fallback path for incremental inference or non-chunk-aligned sequences.
        if cache is None:
            cache = self.alloc_cache(batch_size=batch_size)
        step_logits = []
        for pos in range(seqlen):
            logits_t, cache = self.step(input_ids[:, pos : pos + 1], cache)
            step_logits.append(logits_t)
        logits = torch.cat(step_logits, dim=1)
        return logits, cache

    def generate(
        self,
        input_ids: LongTensor,
        max_new_length: int = 20,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        eos_token_id: int = 0,
    ) -> Iterable[tuple[int, MCInferenceCache]]:
        cache = self.alloc_cache(batch_size=1)
        prefix = input_ids[:-1]
        tokens = input_ids[-1:].unsqueeze(0)

        for i in range(prefix.shape[0]):
            _, cache = self.step(prefix[i : i + 1].unsqueeze(0), cache)

        for _ in range(max_new_length):
            with torch.no_grad():
                out, cache = self.step(tokens, cache)

            logits = out[0, -1]
            if temperature != 1.0:
                logits = logits / temperature
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, k=top_k)[0][-1]
                logits[indices_to_remove] = -torch.inf
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cum_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = -torch.inf

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            if next_token.item() == eos_token_id:
                return
            tokens = next_token.unsqueeze(0)
            yield cast(int, next_token.item()), cache
