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
"""Beam search decoding control.

This module owns the *control half* of beam search: given the per-beam top-W next
tokens and their log-probabilities (produced by
:func:`sglang.srt.layers.sampler.beam_sample_topk`), it decides which W hypotheses
survive each step, tracks cumulative scores, routes EOS-terminated hypotheses into a
"done" pool, and selects the final top-``n`` hypotheses.

The algorithm follows HuggingFace ``transformers`` beam search semantics so it can be
validated against an independent reference implementation:

* Running ``beam_scores`` are raw cumulative log-probabilities.
* Each step considers ``2 * W`` candidates (which guarantees at least W non-EOS
  continuations survive even if some best candidates are EOS).
* The length penalty (``score / generated_len ** length_penalty``) is applied **only**
  when ranking finished hypotheses and at finalization — never to the running score
  (otherwise it would be applied repeatedly and double-count).

The single most important output for the rest of the engine is the **back-pointer**
(:attr:`BeamSearchStepOutput.parent_ptr`): for each of the W surviving beams it names
which previous beam it continues. The KV-cache / attention layer consumes this to make
beam ``j`` inherit beam ``i``'s key/value history before the next forward pass.

This module is intentionally dependency-light (``torch`` + stdlib only) so it can be
unit-tested on CPU without bringing up the scheduler or the model runner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import torch


@dataclass
class BeamSearchStepOutput:
    """Result of advancing a beam group by one decode step.

    All tensors are 1-D of length ``W`` (the beam width) and group-local.
    """

    next_tokens: torch.Tensor  # [W] int64 — token appended to each surviving beam
    parent_ptr: torch.Tensor  # [W] int64 — which previous beam each survivor continues
    next_scores: torch.Tensor  # [W] float — cumulative raw log-prob of each survivor


class BeamHypotheses:
    """Pool of the best finished (EOS-terminated) hypotheses for one request.

    Mirrors ``transformers.generation.beam_search.BeamHypotheses``. Hypotheses are
    ranked by length-penalized score; the pool keeps at most ``num_beams`` of them.
    """

    def __init__(
        self,
        num_beams: int,
        length_penalty: float,
        early_stopping: Union[bool, str],
        max_len: Optional[int] = None,
    ):
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.max_len = max_len
        self.beams: List[Tuple[float, List[int]]] = []
        self.worst_score = 1e9

    def __len__(self) -> int:
        return len(self.beams)

    def add(self, tokens: List[int], sum_logprobs: float, generated_len: int) -> None:
        """Add a completed hypothesis, evicting the worst if the pool overflows."""
        score = sum_logprobs / (generated_len**self.length_penalty)
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, list(tokens)))
            if len(self) > self.num_beams:
                # Drop the lowest-scoring hypothesis and update the new worst score.
                sorted_scores = sorted(
                    (s, idx) for idx, (s, _) in enumerate(self.beams)
                )
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs: float, cur_len: int) -> bool:
        """Whether no future hypothesis can beat what's already in the pool.

        ``best_sum_logprobs`` is the highest cumulative (raw) score among the current
        running candidates; ``cur_len`` is the number of tokens generated so far.
        """
        if len(self) < self.num_beams:
            return False
        if self.early_stopping is True:
            return True
        if self.early_stopping == "never":
            # The best attainable future score depends on whether the length penalty
            # rewards (>0) or punishes (<0) longer sequences.
            if self.length_penalty > 0.0:
                assert self.max_len is not None, "max_len required for early_stopping='never'"
                ref_len = self.max_len
            else:
                ref_len = cur_len
            highest_attainable = best_sum_logprobs / (ref_len**self.length_penalty)
        else:  # early_stopping is False — the canonical heuristic
            highest_attainable = best_sum_logprobs / (cur_len**self.length_penalty)
        return self.worst_score >= highest_attainable


class BeamGroup:
    """Per-request beam-search state (W live hypotheses + a finished pool)."""

    def __init__(
        self,
        rid: str,
        beam_width: int,
        num_return: int,
        eos_token_ids: Iterable[int],
        length_penalty: float = 1.0,
        early_stopping: Union[bool, str] = False,
        max_new_tokens: int = 128,
        device: Union[str, torch.device] = "cpu",
        score_dtype: torch.dtype = torch.float32,
    ):
        assert beam_width >= 1
        assert 1 <= num_return <= beam_width
        self.rid = rid
        self.W = beam_width
        self.num_return = num_return
        self.eos_token_ids: Set[int] = set(int(t) for t in eos_token_ids)
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.max_new_tokens = max_new_tokens
        self.device = torch.device(device)
        self.score_dtype = score_dtype

        # Only beam 0 is "live" at the start; the others are masked with -inf so the
        # first step expands the single prefill state into W distinct beams, all with
        # parent_ptr == 0 (the fan-out).
        self.beam_scores = torch.full(
            (self.W,), float("-inf"), dtype=score_dtype, device=self.device
        )
        self.beam_scores[0] = 0.0
        self.beam_tokens: List[List[int]] = [[] for _ in range(self.W)]

        self.hyps = BeamHypotheses(
            num_beams=self.W,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            max_len=max_new_tokens,
        )
        self.cur_len = 0
        self.done = False

    @property
    def num_candidates(self) -> int:
        """Candidates the sampler must return per beam (``2 * W`` per HF)."""
        return 2 * self.W

    def is_finished(self) -> bool:
        return self.done or self.cur_len >= self.max_new_tokens

    def step(
        self, topk_tokens: torch.Tensor, topk_logprobs: torch.Tensor
    ) -> BeamSearchStepOutput:
        """Advance one decode step.

        Args:
            topk_tokens: ``[W, C]`` int — per-beam candidate token ids (C >= 2W).
            topk_logprobs: ``[W, C]`` float — their log-probabilities.

        Returns:
            :class:`BeamSearchStepOutput` with the W surviving beams' tokens, their
            back-pointers, and cumulative scores.
        """
        W, C = topk_tokens.shape
        assert W == self.W, f"expected {self.W} beam rows, got {W}"
        assert C >= 2 * W, f"need >= {2 * W} candidates per beam, got {C}"

        # Cumulative score of each (beam, candidate) pair.
        cand_scores = self.beam_scores[:, None] + topk_logprobs.to(self.score_dtype)
        flat_scores = cand_scores.reshape(-1)
        flat_tokens = topk_tokens.reshape(-1)
        flat_parent = torch.arange(W, device=self.device).repeat_interleave(C)

        n_sel = min(2 * W, flat_scores.numel())
        top_scores, top_idx = torch.topk(flat_scores, n_sel)
        sel_tokens = flat_tokens[top_idx].tolist()
        sel_parents = flat_parent[top_idx].tolist()
        sel_scores = top_scores.tolist()

        self.cur_len += 1

        next_tokens: List[int] = []
        next_parents: List[int] = []
        next_scores: List[float] = []
        for rank in range(n_sel):
            token = int(sel_tokens[rank])
            parent = int(sel_parents[rank])
            score = float(sel_scores[rank])
            if token in self.eos_token_ids:
                # EOS ranked beyond the top-W can never enter the kept set, so HF
                # ignores it; otherwise the completed hypothesis joins the pool.
                if rank >= W:
                    continue
                self.hyps.add(
                    self.beam_tokens[parent] + [token], score, self.cur_len
                )
            else:
                next_tokens.append(token)
                next_parents.append(parent)
                next_scores.append(score)
            if len(next_tokens) == W:
                break

        # With C >= 2W the top-2W candidates always contain >= W non-EOS tokens, so the
        # loop fills all W slots. Guard defensively against pathological inputs.
        assert len(next_tokens) == W, (
            "beam search could not find W non-EOS continuations; "
            "ensure the sampler returns >= 2W candidates per beam."
        )

        new_beam_tokens = [
            self.beam_tokens[next_parents[i]] + [next_tokens[i]] for i in range(W)
        ]
        self.beam_tokens = new_beam_tokens
        self.beam_scores = torch.tensor(
            next_scores, dtype=self.score_dtype, device=self.device
        )

        # Update termination: the best running candidate's raw score bounds the future.
        best_running = float(sel_scores[0])
        if self.hyps.is_done(best_running, self.cur_len):
            self.done = True

        return BeamSearchStepOutput(
            next_tokens=torch.tensor(
                next_tokens, dtype=torch.int64, device=self.device
            ),
            parent_ptr=torch.tensor(
                next_parents, dtype=torch.int64, device=self.device
            ),
            next_scores=self.beam_scores.clone(),
        )

    def finalize(self) -> List[Tuple[float, List[int]]]:
        """Return the top-``num_return`` hypotheses as ``(penalized_score, tokens)``.

        Sorted by length-penalized score, descending. If the group did not terminate
        via the done-heuristic, the still-running beams are folded into the pool first.
        """
        if not self.done:
            for i in range(self.W):
                self.hyps.add(
                    self.beam_tokens[i], float(self.beam_scores[i]), max(self.cur_len, 1)
                )
        ranked = sorted(self.hyps.beams, key=lambda x: x[0], reverse=True)
        return ranked[: self.num_return]


class BeamSearchManager:
    """Side table of :class:`BeamGroup`s, keyed by request id.

    Lives on the scheduler. Keeping beam state here (rather than on ``Req``) mirrors how
    speculative decoding keeps tree state in ``batch.spec_info`` separate from ``Req``,
    which is reused/merged/retracted throughout the request lifecycle.
    """

    def __init__(self) -> None:
        self.groups: Dict[str, BeamGroup] = {}

    def __contains__(self, rid: str) -> bool:
        return rid in self.groups

    def add(
        self,
        rid: str,
        beam_width: int,
        num_return: int,
        eos_token_ids: Iterable[int],
        length_penalty: float = 1.0,
        early_stopping: Union[bool, str] = False,
        max_new_tokens: int = 128,
        device: Union[str, torch.device] = "cpu",
    ) -> BeamGroup:
        group = BeamGroup(
            rid=rid,
            beam_width=beam_width,
            num_return=num_return,
            eos_token_ids=eos_token_ids,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        self.groups[rid] = group
        return group

    def step(
        self, rid: str, topk_tokens: torch.Tensor, topk_logprobs: torch.Tensor
    ) -> BeamSearchStepOutput:
        return self.groups[rid].step(topk_tokens, topk_logprobs)

    def finalize(self, rid: str) -> List[Tuple[float, List[int]]]:
        return self.groups[rid].finalize()

    def remove(self, rid: str) -> None:
        self.groups.pop(rid, None)
