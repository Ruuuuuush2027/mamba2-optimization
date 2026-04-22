"""
mamba2-mc
=========

Memory Caching (Behrouz et al., 2026, arXiv:2602.24281) applied on top of the
minimal Mamba-2 implementation in `mamba2.py`.

Supports two retrieval variants from the paper:
  - GRM (Gated Residual Memory, default, equivalent to Memory Soup for linear
    memory like Mamba-2): attend to online memory + all cached segments.
  - SSC (Sparse Selective Caching): keep only top-k cached segments per token.

Design notes
------------
* At each segment boundary (every `segment_size` tokens), the current
  `ssm_state` of each layer is cloned and appended to a per-layer cache, along
  with a mean-pooled projection of the segment's input tokens.
* At every decode step, retrieval fuses `online + cached` states with
  gates γ_i = softmax_i(<u_t, pool_i>), then does the usual `C · state` read.
  Since Mamba-2 is linear, we fuse states first and read once (Eq. 13).
* A scalar `online_bias` (init large) biases the softmax toward the online
  segment so that a freshly-initialized MC wrapper matches vanilla Mamba-2.
  During fine-tuning this bias shrinks and the W_u projection learns routing.

Limitations
-----------
* Prefill here is done token-by-token via `step_mc`. This keeps cache
  bookkeeping uniform but is slower than the parallel `ssd` path. A faster
  version would chunk prefill by segment, run `ssd` per segment with
  `initial_states`, and cache `final_state` at each boundary.
"""

import json
from dataclasses import dataclass
from typing import Iterable, NamedTuple, cast

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import LongTensor, Tensor, nn

from mamba2 import Device, Mamba2, Mamba2Config, RMSNorm, silu


@dataclass
class Mamba2MCConfig(Mamba2Config):
    segment_size: int = 256
    d_pool: int = 64
    top_k_ssc: int = 0  # 0 = GRM (attend all cached); >0 = SSC top-k
    online_bias_init: float = 20.0  # init bias on online segment's logit (20 → γ_cached ≈ 2e-9 at init)


class MCLayerCache(NamedTuple):
    conv_state: Tensor              # (B, d_inner + 2*d_state, d_conv)
    ssm_state: Tensor               # (B, H, P, N) — online memory
    cached_states: list[Tensor]     # each (B, H, P, N)
    cached_pools: list[Tensor]      # each (B, d_pool)
    seg_pool_sum: Tensor            # (B, d_pool) — running sum for current seg
    seg_pool_count: int             # tokens accumulated into seg_pool_sum


def _alloc_layer_cache(args: Mamba2MCConfig, batch_size: int, device: Device) -> MCLayerCache:
    return MCLayerCache(
        conv_state=torch.zeros(
            batch_size, args.d_inner + 2 * args.d_state, args.d_conv, device=device
        ),
        ssm_state=torch.zeros(
            batch_size, args.nheads, args.headdim, args.d_state, device=device
        ),
        cached_states=[],
        cached_pools=[],
        seg_pool_sum=torch.zeros(batch_size, args.d_pool, device=device),
        seg_pool_count=0,
    )


class Mamba2MC(Mamba2):
    """Mamba-2 mixer with Memory Caching retrieval."""

    def __init__(self, args: Mamba2MCConfig, device: Device = None):
        super().__init__(args, device=device)
        self.args: Mamba2MCConfig = args
        self.W_u = nn.Linear(args.d_model, args.d_pool, bias=False, device=device)
        # Large positive bias → fresh model weights online ~1, behaves like vanilla Mamba-2
        self.online_bias = nn.Parameter(
            torch.full((), args.online_bias_init, device=device)
        )
        # Zero-init W_u so that without training, dot products are 0 and only the bias decides γ
        nn.init.zeros_(self.W_u.weight)

    def step_mc(self, u: Tensor, h: MCLayerCache) -> tuple[Tensor, MCLayerCache]:
        """Single-token step with MC retrieval. Replaces `Mamba2.step`."""
        assert u.shape[1] == 1, "Only one token can be decoded per MC step"

        u_sq = u.squeeze(1)                    # (B, D)
        u_proj = self.W_u(u_sq)                # (B, d_pool)

        zxbcdt = self.in_proj(u_sq)
        z, xBC, dt = torch.split(
            zxbcdt,
            [self.args.d_inner, self.args.d_inner + 2 * self.args.d_state, self.args.nheads],
            dim=-1,
        )

        # Advance convolution cache
        h.conv_state.copy_(torch.roll(h.conv_state, shifts=-1, dims=-1))
        h.conv_state[:, :, -1] = xBC
        xBC = torch.sum(
            h.conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
        )
        xBC += self.conv1d.bias
        xBC = silu(xBC)

        x, B, C = torch.split(
            xBC, [self.args.d_inner, self.args.d_state, self.args.d_state], dim=-1
        )
        A = -torch.exp(self.A_log)

        # Update online SSM state (in-place on h.ssm_state)
        dt = F.softplus(dt + self.dt_bias)
        dA = torch.exp(dt * A)
        x = rearrange(x, "b (h p) -> b h p", p=self.args.headdim)
        dBx = torch.einsum("bh, bn, bhp -> bhpn", dt, B, x)
        h.ssm_state.copy_(h.ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)

        # ---- MC retrieval: fuse online + cached states, then read with C ----
        if len(h.cached_states) > 0:
            cached_pools = torch.stack(h.cached_pools, dim=1)            # (B, S, d_pool)
            online_pool = h.seg_pool_sum / max(h.seg_pool_count, 1)      # (B, d_pool)
            all_pools = torch.cat([cached_pools, online_pool.unsqueeze(1)], dim=1)  # (B, S+1, d_pool)

            scores = torch.einsum("bd,bsd->bs", u_proj, all_pools)       # (B, S+1)
            # Bias online slot (last) so fresh model behaves like vanilla Mamba-2
            online_bias_vec = torch.zeros_like(scores)
            online_bias_vec[:, -1] = self.online_bias
            scores = scores + online_bias_vec

            if self.args.top_k_ssc > 0 and len(h.cached_states) > self.args.top_k_ssc:
                k = self.args.top_k_ssc
                cached_scores = scores[:, :-1]
                _, topk_idx = torch.topk(cached_scores, k, dim=-1)
                mask = torch.full_like(scores, float("-inf"))
                mask.scatter_(-1, topk_idx, 0.0)
                mask[:, -1] = 0.0  # always keep online
                scores = scores + mask

            gammas = F.softmax(scores, dim=-1)                           # (B, S+1)

            cached_stack = torch.stack(h.cached_states, dim=1)           # (B, S, H, P, N)
            all_states = torch.cat(
                [cached_stack, h.ssm_state.unsqueeze(1)], dim=1
            )                                                             # (B, S+1, H, P, N)
            fused_state = torch.einsum("bs,bshpn->bhpn", gammas, all_states)
            y = torch.einsum("bhpn, bn -> bhp", fused_state, C)
        else:
            y = torch.einsum("bhpn, bn -> bhp", h.ssm_state, C)

        y = y + rearrange(self.D, "h -> h 1") * x
        y = rearrange(y, "b h p -> b (h p)")
        y = self.norm(y, z)
        y = self.out_proj(y)

        # ---- Update segment pool; cache & reset at segment boundary ----
        new_pool_sum = h.seg_pool_sum + u_proj
        new_count = h.seg_pool_count + 1

        if new_count >= self.args.segment_size:
            h = h._replace(
                cached_states=h.cached_states + [h.ssm_state.clone()],
                cached_pools=h.cached_pools + [new_pool_sum / new_count],
                seg_pool_sum=torch.zeros_like(new_pool_sum),
                seg_pool_count=0,
            )
        else:
            h = h._replace(seg_pool_sum=new_pool_sum, seg_pool_count=new_count)

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
        online_bias_init: float = 20.0,
    ) -> "Mamba2MCLMHeadModel":
        """Load a pretrained Mamba-2 and wrap with MC; MC params are freshly initialized."""
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
            d_pool=d_pool,
            top_k_ssc=top_k_ssc,
            online_bias_init=online_bias_init,
        )

        map_location = "cpu" if device is None else device
        state_dict = torch.load(
            state_dict_path, weights_only=True, map_location=map_location, mmap=True
        )
        model = Mamba2MCLMHeadModel(args, device=device)
        # strict=False: MC-only params (W_u, online_bias) stay at their initial values
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        assert len(unexpected) == 0, f"Unexpected keys in pretrained state dict: {unexpected}"
        # Verify only MC params are missing
        allowed_missing = {"W_u.weight", "online_bias"}
        for k in missing:
            leaf = k.rsplit(".", 1)[-1]
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

        logits_steps: list[Tensor] = []
        for t in range(seqlen):
            step_logits, caches = self.step(input_ids[:, t : t + 1], caches)
            logits_steps.append(step_logits)

        logits = torch.cat(logits_steps, dim=1)
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

        # Prefill token-by-token so MC cache is built consistently
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
