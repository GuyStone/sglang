# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Two-level cascade attention primitive.

Cascade attention computes attention over a **shared prefix** of keys/values once and
over each query's **unique suffix** separately, then merges the two partial results with
the log-sum-exp (online-softmax) identity. This is exactly the access pattern of beam
search: the W beams of a request share a long common prefix (prompt + agreed tokens) and
diverge only in their short tails, so the expensive prefix attention is done once instead
of W times.

SGLang already performs this two-pass + merge inline in several attention backends (for
speculative decoding's ``target_verify`` with ``topk > 1``). This module factors out the
two reusable pieces so beam search and speculative decoding can share them rather than
duplicating the pattern:

* :func:`merge_attn_states` — the pure-PyTorch LSE merge (a backend-agnostic reference
  and CPU fallback for ``sgl_kernel``'s ``merge_state_v2`` / FlashInfer's ``merge_state``).
* :func:`cascade_attend` — the orchestration: attend the shared prefix once, attend each
  suffix, merge.

The functions take/return outputs in the canonical ``merge_state`` layout: attention
output ``[N, H, D]`` and log-sum-exp ``[N, H]`` (N = number of queries, H = heads,
D = head dim). Backends are responsible for any transpose between their kernel's lse
layout and this one (e.g. FlashAttention returns lse as ``[H, N]`` and transposes before
merging).
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch

AttnResult = Tuple[torch.Tensor, torch.Tensor]  # (out [N, H, D], lse [N, H])
MergeFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], AttnResult
]


def merge_attn_states(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
) -> AttnResult:
    """Log-sum-exp merge of two attention partitions (pure PyTorch).

    Combines attention computed over two disjoint sets of keys (e.g. a shared prefix and
    a per-beam suffix) into the result of attention over their union, using the
    online-softmax identity::

        w_a = exp(lse_a - m),  w_b = exp(lse_b - m),  m = max(lse_a, lse_b)
        out = (w_a * out_a + w_b * out_b) / (w_a + w_b)
        lse = m + log(w_a + w_b)

    This is the math behind FlashInfer's ``merge_state`` and ``sgl_kernel``'s
    ``merge_state_v2``; it is provided here as a backend-agnostic reference and a CPU /
    unsupported-dtype fallback.

    Args:
        out_a, out_b: ``[N, H, D]`` attention outputs over partitions A and B.
        lse_a, lse_b: ``[N, H]`` log-sum-exp of the attention scores per partition.

    Returns:
        ``(out, lse)`` — merged output ``[N, H, D]`` and merged lse ``[N, H]``.
    """
    max_lse = torch.maximum(lse_a, lse_b)
    # A query may attend to nothing in one partition (lse = -inf); keep the combine
    # finite. If both partitions are empty the result is undefined, but stays NaN-free.
    max_lse = torch.nan_to_num(max_lse, neginf=0.0)
    w_a = torch.exp(lse_a - max_lse)
    w_b = torch.exp(lse_b - max_lse)
    denom = (w_a + w_b).clamp_min(1e-20)
    out = (w_a.unsqueeze(-1) * out_a + w_b.unsqueeze(-1) * out_b) / denom.unsqueeze(-1)
    lse = max_lse + torch.log(denom)
    return out, lse


def _default_merge_fn() -> MergeFn:
    """Resolve the best available merge implementation.

    Prefers ``sgl_kernel``/FlashInfer ``merge_state`` (fused CUDA kernel) and falls back
    to the pure-PyTorch :func:`merge_attn_states` when it is unavailable (e.g. CPU).
    """
    try:
        from sglang.srt.layers.attention.merge_state import merge_state

        return merge_state
    except Exception:
        return merge_attn_states


def cascade_attend(
    prefix_attn: Callable[[], AttnResult],
    suffix_attn: Callable[[], AttnResult],
    merge_fn: Optional[MergeFn] = None,
) -> AttnResult:
    """Two-level cascade attention: shared prefix attended once, suffix per query, merged.

    Args:
        prefix_attn: callable returning ``(out_p [N, H, D], lse_p [N, H])`` — attention
            of the N queries over the **shared** prefix KV. For beam search the
            ``N = num_groups * W`` queries of a group all attend the *same* prefix, so
            the backend computes this with a single kernel launch.
        suffix_attn: callable returning ``(out_s [N, H, D], lse_s [N, H])`` — attention
            of each query over its **own** divergent tail KV.
        merge_fn: ``(out_a, lse_a, out_b, lse_b) -> (out, lse)``. Defaults to the fused
            ``merge_state`` kernel when available, else the pure-PyTorch reference.

    Returns:
        ``(out [N, H, D], lse [N, H])`` — attention over prefix ∪ suffix.
    """
    if merge_fn is None:
        merge_fn = _default_merge_fn()
    out_p, lse_p = prefix_attn()
    out_s, lse_s = suffix_attn()
    return merge_fn(out_p, lse_p, out_s, lse_s)
