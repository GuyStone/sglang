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
"""Runtime glue for beam search decoding: per-step orchestration + KV reorder.

This sits between the beam *control* logic (:mod:`sglang.srt.managers.beam_search`,
which decides which W hypotheses survive and emits a per-group ``parent_ptr``) and the
KV-cache / scheduler. It owns two things:

1. :func:`run_beam_decode_step` — slices the batched sampler top-W output per request
   group, advances each :class:`BeamGroup`, and assembles the global step result
   (next tokens, ``parent_ptr``, scores, per-group finished flags, reorder plan).

2. The **KV reorder**. Each step, surviving beam ``j`` of a group continues parent beam
   ``p = parent_ptr[j]``. Because the beams' token history up to this step is identical
   in *content* (they only diverge in the token chosen *now*, whose KV is computed on the
   next forward), beam ``j`` can inherit parent ``p``'s history by **repointing its
   ``req_to_token`` row to parent ``p``'s row** — no KV *data* copy is needed. The
   historical KV slots are shared read-only across beams (exactly like radix prefix
   sharing); only the next token gets a fresh slot. This is cheaper than copying tail KV.

   :func:`compute_beam_row_reorder` computes the pure index plan (which source row each
   surviving row inherits, which rows are unchanged, and which beams' just-written token
   slot became orphaned and can be freed). :func:`apply_beam_row_reorder` applies it to a
   ``req_to_token`` tensor with a hazard-free gather.

Layout convention: a batch holds ``num_groups`` beam-search requests, each materialized as
``W`` contiguous rows. Global row ``g * W + j`` is beam ``j`` of group ``g``. ``parent_ptr``
is group-local (values in ``[0, W)``); the global source row of survivor ``g*W+j`` is
``g*W + parent_ptr[g, j]``.

The pure index/bookkeeping functions here are unit-tested on CPU
(``test/srt/cpu/test_beam_search_cascade.py``); the scheduler/attention hooks that call
them are documented in ``benchmark/beam_search_cascade/INTEGRATION.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from sglang.srt.managers.beam_search import BeamSearchManager, BeamSearchStepOutput


@dataclass
class BeamDecodeInput:
    """Beam geometry carried on ``ForwardBatch.spec_info`` for cascade attention.

    Consumed (duck-typed) by ``FlashAttentionBackend._init_beam_decode_metadata`` to build
    the two-pass cascade metadata. Field names are part of that contract.

    For ``num_groups`` requests each with ``beam_width`` beams:
      * ``prefix_lens`` ``[num_groups]`` — shared committed prefix length ``D_g`` per group
        (the longest-common-ancestor length of the group's W beams).
      * ``prefix_page_table`` ``[num_groups, max_prefix]`` int32 — the shared prefix's KV
        slots per group (pass 1 reads this once for all W beams of the group).
      * ``tail_lens`` ``[num_groups*W]`` int32 — divergent-tail length ``L - D_g`` per beam.
      * ``tail_page_table`` ``[num_groups*W, max_tail]`` int32 — each beam's tail KV slots
        (pass 2).
    """

    beam_width: int
    prefix_lens: torch.Tensor
    prefix_page_table: torch.Tensor
    tail_lens: torch.Tensor
    tail_page_table: torch.Tensor


@dataclass
class BeamReorderPlan:
    """Index plan to rewrite per-beam ``req_to_token`` rows after a beam step.

    All row indices are *global* (``g * W + beam``).
    """

    src_rows: torch.Tensor  # [num_groups*W] int64 — source row each survivor inherits
    in_place: torch.Tensor  # [num_groups*W] bool — True where parent == self (no move)
    free_beam_rows: torch.Tensor  # [k] int64 — beams not chosen as any survivor's parent;
    #                               their just-written token slot is orphaned (free it)


def compute_beam_row_reorder(
    parent_ptr: torch.Tensor, beam_width: int
) -> BeamReorderPlan:
    """Compute the KV row-repoint plan from per-group back-pointers.

    Args:
        parent_ptr: ``[num_groups, beam_width]`` int — group-local parent index in
            ``[0, beam_width)`` for each surviving beam.
        beam_width: W.

    Returns:
        :class:`BeamReorderPlan` with global ``src_rows``, ``in_place`` mask, and
        ``free_beam_rows`` (beams referenced by no survivor → their owned new-token slot
        is orphaned).
    """
    assert parent_ptr.dim() == 2 and parent_ptr.shape[1] == beam_width
    num_groups = parent_ptr.shape[0]
    device = parent_ptr.device
    W = beam_width

    group_offset = (torch.arange(num_groups, device=device) * W).unsqueeze(1)  # [G,1]
    src_rows = (group_offset + parent_ptr).reshape(-1).to(torch.int64)  # [G*W]

    local_idx = torch.arange(W, device=device).unsqueeze(0)  # [1,W]
    in_place = (parent_ptr == local_idx).reshape(-1)  # [G*W] bool

    # A beam is "referenced" if some survivor in its group continues from it.
    referenced = torch.zeros(num_groups, W, dtype=torch.bool, device=device)
    referenced.scatter_(1, parent_ptr.to(torch.int64), True)
    free_beam_rows = (~referenced).reshape(-1).nonzero(as_tuple=True)[0].to(torch.int64)

    return BeamReorderPlan(
        src_rows=src_rows, in_place=in_place, free_beam_rows=free_beam_rows
    )


def apply_beam_row_reorder(
    req_to_token: torch.Tensor,
    beam_pool_indices: torch.Tensor,
    plan: BeamReorderPlan,
    history_len: int,
) -> None:
    """Repoint each beam's ``req_to_token`` row to its parent's, in place.

    ``new_row[i][:history_len] = old_row[src_rows[i]][:history_len]`` for all i, done via
    a clone-then-scatter so survivors that share a parent all read the original rows
    (no read-after-write hazard).

    Args:
        req_to_token: ``[pool_size, max_context_len]`` token→KV-slot map (mutated).
        beam_pool_indices: ``[num_groups*W]`` int — the ``req_pool_idx`` of each beam row,
            indexed by global beam row.
        plan: from :func:`compute_beam_row_reorder`.
        history_len: number of valid token positions to carry over (L).
    """
    src_pool = beam_pool_indices[plan.src_rows]  # [G*W] pool rows to read from
    gathered = req_to_token[src_pool, :history_len].clone()  # hazard-free snapshot
    req_to_token[beam_pool_indices, :history_len] = gathered


def orphan_free_slots(
    plan: BeamReorderPlan, out_cache_loc: torch.Tensor
) -> torch.Tensor:
    """KV slots that can be freed after the step.

    Each beam wrote its current token's K/V to ``out_cache_loc[global_beam_row]`` this
    step. Beams that no survivor inherits (``plan.free_beam_rows``) own a slot that is now
    unreferenced and may be returned to the allocator.
    """
    if plan.free_beam_rows.numel() == 0:
        return out_cache_loc[:0]
    return out_cache_loc[plan.free_beam_rows]


@dataclass
class BeamDecodeStepResult:
    next_tokens: torch.Tensor  # [num_groups*W] int64 — token to feed next forward
    parent_ptr: torch.Tensor  # [num_groups, W] int64 — group-local back-pointers
    next_scores: torch.Tensor  # [num_groups*W] float — cumulative beam scores
    finished: List[bool]  # per-group: group reached a termination condition
    reorder_plan: BeamReorderPlan


def run_beam_decode_step(
    manager: BeamSearchManager,
    group_rids: Sequence[str],
    topk_tokens: torch.Tensor,
    topk_logprobs: torch.Tensor,
    beam_width: int,
) -> BeamDecodeStepResult:
    """Advance every beam group one decode step and assemble the global step result.

    Args:
        manager: holds a :class:`BeamGroup` per ``rid``.
        group_rids: rids in batch order; group ``g`` occupies global rows
            ``[g*W, (g+1)*W)``.
        topk_tokens: ``[num_groups*W, C]`` int — per-beam candidate tokens (C >= 2W).
        topk_logprobs: ``[num_groups*W, C]`` float — their log-probs.
        beam_width: W (assumed uniform across the batch's beam groups).

    Returns:
        :class:`BeamDecodeStepResult`.
    """
    W = beam_width
    num_groups = len(group_rids)
    assert topk_tokens.shape[0] == num_groups * W, (
        f"expected {num_groups * W} rows, got {topk_tokens.shape[0]}"
    )

    next_tokens = torch.empty(num_groups * W, dtype=torch.int64, device=topk_tokens.device)
    parent_ptr = torch.empty(num_groups, W, dtype=torch.int64, device=topk_tokens.device)
    next_scores = torch.empty(num_groups * W, dtype=torch.float32, device=topk_tokens.device)
    finished: List[bool] = []

    for g, rid in enumerate(group_rids):
        rows = slice(g * W, (g + 1) * W)
        out: BeamSearchStepOutput = manager.step(
            rid, topk_tokens[rows], topk_logprobs[rows]
        )
        next_tokens[rows] = out.next_tokens.to(next_tokens.dtype)
        parent_ptr[g] = out.parent_ptr.to(parent_ptr.dtype)
        next_scores[rows] = out.next_scores.to(next_scores.dtype)
        finished.append(manager.groups[rid].is_finished())

    plan = compute_beam_row_reorder(parent_ptr, W)
    return BeamDecodeStepResult(
        next_tokens=next_tokens,
        parent_ptr=parent_ptr,
        next_scores=next_scores,
        finished=finished,
        reorder_plan=plan,
    )
