"""
mamba2-mc
=========

Efficient memory caching for Mamba-2:
* Cache only recent SSM states at segment boundaries.
* Fuse cached states + current online SSM state using a learnable weight vector W.
* Read once with C from the fused state.
"""

import json
from dataclasses import dataclass
from typing import Iterable, cast

import torch
import torch.nn.functional as F
from torch import LongTensor, Tensor, nn

from mamba2 import Device, Mamba2, Mamba2Config, RMSNorm, silu


@dataclass
class Mamba2MCConfig(Mamba2Config):
    segment_size: int = 256
    max_cached_segments: int = 8  # keep only latest X cached segment states
    detach_cached_segments: bool = True
    # Backward-compatible no-op fields from previous MC implementation.
    d_pool: int = 64
    top_k_ssc: int = 0
    online_bias_init: float = 20.0


@dataclass
class MCLayerCache:
    conv_state: Tensor              # (B, d_inner + 2*d_state, d_conv)
    ssm_state: Tensor               # (B, H, P, N) online memory
    cached_states: Tensor | None    # (B, Smax, H, P, N) ring buffer
    cached_count: int               # number of valid cached segments
    cached_next_idx: int            # next write index in ring buffer
    segment_tokens: int             # tokens accumulated in current segment


def _alloc_layer_cache(args: Mamba2MCConfig, batch_size: int, device: Device) -> MCLayerCache:
    return MCLayerCache(
        conv_state=torch.zeros(
            batch_size, args.d_inner + 2 * args.d_state, args.d_conv, device=device
        ),
        ssm_state=torch.zeros(
            batch_size, args.nheads, args.headdim, args.d_state, device=device
        ),
        cached_states=None,
        cached_count=0,
        cached_next_idx=0,
        segment_tokens=0,
    )


class Mamba2MC(Mamba2):
    """Mamba-2 mixer with efficient state-only memory caching."""

    def __init__(self, args: Mamba2MCConfig, device: Device = None):
        super().__init__(args, device=device)
        self.args: Mamba2MCConfig = args
        if self.args.max_cached_segments <= 0:
            raise ValueError("max_cached_segments must be > 0 for state-only MC.")

        # W has one logit per cache slot and one for the online state.
        # Weights are softmax-normalized on the active subset each step.
        self.W = nn.Parameter(torch.zeros(args.max_cached_segments + 1, device=device))
        with torch.no_grad():
            self.W[-1] = 5.0  # start close to vanilla Mamba behavior (favor online state)

    def _ordered_cached_states(self, h: MCLayerCache) -> Tensor | None:
        if h.cached_states is None or h.cached_count == 0:
            return None

        if h.cached_count < self.args.max_cached_segments:
            return h.cached_states[:, : h.cached_count]

        # Ring buffer is full: reorder to chronological [oldest ... newest].
        start = h.cached_next_idx
        if start == 0:
            return h.cached_states
        return torch.cat([h.cached_states[:, start:], h.cached_states[:, :start]], dim=1)

    def _append_cached_state(self, h: MCLayerCache, state: Tensor) -> None:
        m = self.args.max_cached_segments
        if h.cached_states is None:
            h.cached_states = torch.empty(
                state.shape[0],
                m,
                state.shape[1],
                state.shape[2],
                state.shape[3],
                device=state.device,
                dtype=state.dtype,
            )

        h.cached_states[:, h.cached_next_idx] = state
        h.cached_next_idx = (h.cached_next_idx + 1) % m
        h.cached_count = min(h.cached_count + 1, m)

    def _active_weights(self, n_cached: int, dtype: torch.dtype, device: torch.device) -> Tensor:
        # Cached slots use the most-recent n_cached logits from W[:-1], online uses W[-1].
        cached_logits = self.W[:-1][-n_cached:] if n_cached > 0 else self.W[:-1][:0]
        logits = torch.cat([cached_logits, self.W[-1:].to(cached_logits.device)], dim=0)
        return F.softmax(logits, dim=0).to(device=device, dtype=dtype)

    def step_mc(self, u: Tensor, h: MCLayerCache) -> tuple[Tensor, MCLayerCache]:
        """Single-token step with state-only memory caching."""
        assert u.shape[1] == 1, "Only one token can be decoded per MC step"
        training_mode = self.training

        u_sq = u.squeeze(1)  # (B, D)
        zxbcdt = self.in_proj(u_sq)
        z, xBC, dt = torch.split(
            zxbcdt,
            [self.args.d_inner, self.args.d_inner + 2 * self.args.d_state, self.args.nheads],
            dim=-1,
        )

        if training_mode:
            conv_state = torch.empty_like(h.conv_state)
            conv_state[:, :, :-1] = h.conv_state[:, :, 1:]
            conv_state[:, :, -1] = xBC
        else:
            h.conv_state[:, :, :-1] = h.conv_state[:, :, 1:]
            h.conv_state[:, :, -1] = xBC
            conv_state = h.conv_state

        conv_weight = self.conv1d.weight.squeeze(1)
        xBC = torch.sum(conv_state * conv_weight.unsqueeze(0), dim=-1)
        xBC += self.conv1d.bias
        xBC = silu(xBC)

        x, B, C = torch.split(
            xBC, [self.args.d_inner, self.args.d_state, self.args.d_state], dim=-1
        )
        A = -torch.exp(self.A_log)

        dt = F.softplus(dt + self.dt_bias)
        dA = torch.exp(dt * A)
        x = x.reshape(x.shape[0], self.args.nheads, self.args.headdim)
        dBx = torch.einsum("bh, bn, bhp -> bhpn", dt, B, x)
        if training_mode:
            ssm_state = h.ssm_state * dA.unsqueeze(-1).unsqueeze(-1) + dBx
        else:
            h.ssm_state.copy_(h.ssm_state * dA.unsqueeze(-1).unsqueeze(-1) + dBx)
            ssm_state = h.ssm_state

        cached_states = self._ordered_cached_states(h)
        if cached_states is not None and cached_states.shape[1] > 0:
            n_cached = cached_states.shape[1]
            weights = self._active_weights(n_cached, dtype=ssm_state.dtype, device=ssm_state.device)
            cached_weights = weights[:-1].view(1, n_cached, 1, 1, 1)
            fused_state = torch.sum(cached_states * cached_weights, dim=1)
            fused_state = fused_state + weights[-1] * ssm_state
            y = torch.einsum("bhpn, bn -> bhp", fused_state, C)
        else:
            y = torch.einsum("bhpn, bn -> bhp", ssm_state, C)

        y = y + self.D.unsqueeze(-1) * x
        y = y.reshape(y.shape[0], -1)
        y = self.norm(y, z)
        y = self.out_proj(y)

        new_segment_tokens = h.segment_tokens + 1
        if new_segment_tokens >= self.args.segment_size:
            state_to_store = ssm_state.detach().clone() if self.args.detach_cached_segments else ssm_state.clone()
            self._append_cached_state(h, state_to_store)
            h.segment_tokens = 0
        else:
            h.segment_tokens = new_segment_tokens

        h.conv_state = conv_state
        h.ssm_state = ssm_state
        return y.unsqueeze(1), h


class Mamba2MCLMHeadModel(nn.Module):
    """Top-level LM with Memory Caching Mamba-2 mixers."""

    def __init__(self, args: Mamba2MCConfig, device: Device = None):
        super().__init__()
        self.args = args
        self.device = device

        self.backbone = nn.ModuleDict(
            dict(
                embedding=nn.Embedding(args.vocab_size, args.d_model, device=device),
                layers=nn.ModuleList(
                    [
                        nn.ModuleDict(
                            dict(
                                mixer=Mamba2MC(args, device=device),
                                norm=RMSNorm(args.d_model, device=device),
                            )
                        )
                        for _ in range(args.n_layer)
                    ]
                ),
                norm_f=RMSNorm(args.d_model, device=device),
            )
        )
        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False, device=device)
        self.lm_head.weight = self.backbone.embedding.weight

    @staticmethod
    def from_pretrained(
        huggingface_model_id: str,
        device: Device = None,
        cache_dir: str = None,
        segment_size: int = 256,
        d_pool: int = 64,
        top_k_ssc: int = 0,
        max_cached_segments: int = 8,
        detach_cached_segments: bool = True,
        online_bias_init: float = 20.0,
    ) -> "Mamba2MCLMHeadModel":
        """
        Load a pretrained Mamba-2 and wrap with MC; MC params are freshly initialized.

        `d_pool`, `top_k_ssc`, and `online_bias_init` are kept for API compatibility
        but are unused in the efficient state-only MC implementation.
        """
        _ = (d_pool, top_k_ssc, online_bias_init)

        from transformers.utils import CONFIG_NAME, WEIGHTS_NAME
        from transformers.utils.hub import cached_file

        config_path = cached_file(huggingface_model_id, CONFIG_NAME, cache_dir=cache_dir)
        assert config_path, "Failed to get huggingface config file"
        state_dict_path = cached_file(huggingface_model_id, WEIGHTS_NAME, cache_dir=cache_dir)
        assert state_dict_path, "Failed to get huggingface state dict file"

        config = json.load(open(config_path))
        args = Mamba2MCConfig(
            d_model=config["d_model"],
            n_layer=config["n_layer"],
            vocab_size=config["vocab_size"],
            pad_vocab_size_multiple=config["pad_vocab_size_multiple"],
            segment_size=segment_size,
            max_cached_segments=max_cached_segments,
            detach_cached_segments=detach_cached_segments,
        )

        map_location = "cpu" if device is None else device
        state_dict = torch.load(
            state_dict_path, weights_only=True, map_location=map_location, mmap=True
        )
        model = Mamba2MCLMHeadModel(args, device=device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        assert len(unexpected) == 0, f"Unexpected keys in pretrained state dict: {unexpected}"

        allowed_missing = {"W"}
        for k in missing:
            assert any(k.endswith(a) for a in allowed_missing), f"Unexpected missing key: {k}"

        model.eval()
        return model

    def alloc_cache(self, batch_size: int = 1) -> list[MCLayerCache]:
        return [
            _alloc_layer_cache(self.args, batch_size, self.device)
            for _ in range(self.args.n_layer)
        ]

    def forward(
        self, input_ids: LongTensor, h: list[MCLayerCache] | list[None] | None = None
    ) -> tuple[Tensor, list[MCLayerCache]]:
        """
        Arguments
            input_ids: (batch, seqlen) tokens from `EleutherAI/gpt-neox-20b` tokenizer
            h: hidden states for inference step. If present, `input_ids` should
               have shape (batch, 1) containing the next token.

        Return (logits, h)
            logits: (batch, seqlen, vocab_size)
            h: updated MC cache after processing `input_ids`
        """
        batch_size, seqlen = input_ids.shape

        if h is None:
            caches: list[MCLayerCache] = self.alloc_cache(batch_size)
        else:
            if len(h) != self.args.n_layer:
                raise ValueError(
                    f"Expected {self.args.n_layer} layer caches, got {len(h)}"
                )

            base_caches = self.alloc_cache(batch_size)
            caches = [
                base_caches[i] if layer_cache is None else layer_cache
                for i, layer_cache in enumerate(h)
            ]

        logits: Tensor | None = None
        for t in range(seqlen):
            step_logits, caches = self.step(input_ids[:, t : t + 1], caches)
            if logits is None:
                logits = torch.empty(
                    batch_size,
                    seqlen,
                    step_logits.shape[-1],
                    device=step_logits.device,
                    dtype=step_logits.dtype,
                )
            logits[:, t : t + 1] = step_logits

        assert logits is not None
        return logits[:, :seqlen], caches

    def step(
        self, input_ids: LongTensor, h: list[MCLayerCache]
    ) -> tuple[Tensor, list[MCLayerCache]]:
        """Process a single token per batch with MC."""
        assert input_ids.shape[1] == 1
        x = self.backbone.embedding(input_ids)
        for i, layer in enumerate(self.backbone.layers):
            y, h[i] = layer.mixer.step_mc(layer.norm(x), h[i])
            x = y + x
        x = self.backbone.norm_f(x)
        return self.lm_head(x), h

    def generate(
        self,
        input_ids: LongTensor,
        max_new_length: int = 20,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        eos_token_id: int = 0,
    ) -> Iterable[tuple[int, list[MCLayerCache]]]:
        prefix, tokens = input_ids[:-1], input_ids[-1:].unsqueeze(0)
        h = self.alloc_cache(1)

        for i in range(prefix.shape[0]):
            with torch.no_grad():
                _, h = self.step(prefix[i : i + 1].unsqueeze(0), h)

        for _ in range(max_new_length):
            with torch.no_grad():
                out, h = self.step(tokens, h)
            logits = out[0, -1]
            if temperature != 1.0:
                logits = logits / temperature
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, k=top_k)[0][-1]
                logits[indices_to_remove] = -torch.inf
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cum_probs > 0.5
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = -torch.inf
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            if next_token.item() == eos_token_id:
                return
            tokens = next_token.unsqueeze(0)
            yield cast(int, next_token.item()), h
