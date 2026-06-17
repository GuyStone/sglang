"""CPU unit tests for beam search + cascade attention.

These tests exercise the *control half* of beam search (sampling-param validation, the
``beam_sample_topk`` sampler path, and the ``BeamSearchManager`` selection logic) and the
``cascade_attend`` attention primitive. They run entirely on CPU and do not require a GPU
or a model, so they can be run with::

    .venv/bin/python -m unittest test.srt.cpu.test_beam_search_cascade -v

The beam-search manager is validated against an **independent** pure-Python reference
beam search (different implementation style) driven by a deterministic Markov "model", so
agreement across many seeds is strong evidence of correctness.
"""

import math
import unittest
from typing import List, Optional, Set, Tuple

import torch

from sglang.srt.layers.attention.cascade import cascade_attend, merge_attn_states
from sglang.srt.layers.sampler import beam_sample_topk
from sglang.srt.managers.beam_search import BeamGroup, BeamSearchManager
from sglang.srt.managers.beam_search_runtime import (
    apply_beam_row_reorder,
    compute_beam_row_reorder,
    orphan_free_slots,
    run_beam_decode_step,
)
from sglang.srt.sampling.sampling_params import TOP_K_ALL, SamplingParams

torch.manual_seed(0)


# --------------------------------------------------------------------------------------
# Reference attention (pure torch, returns output + log-sum-exp) for cascade tests.
# --------------------------------------------------------------------------------------
def sdpa_with_lse(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference scaled-dot-product attention returning (out [N,H,D], lse [N,H]).

    q: [N, H, D]; k, v: [N, H, S, D].
    """
    scores = torch.einsum("nhd,nhsd->nhs", q, k) * scale  # [N, H, S]
    lse = torch.logsumexp(scores, dim=-1)  # [N, H]
    attn = torch.softmax(scores, dim=-1)  # [N, H, S]
    out = torch.einsum("nhs,nhsd->nhd", attn, v)  # [N, H, D]
    return out, lse


# --------------------------------------------------------------------------------------
# Independent reference beam search (HF semantics), pure-python loops.
# --------------------------------------------------------------------------------------
def reference_beam_search(
    transition_logits: torch.Tensor,  # [V, V] logits for next token given last token
    init_logits: torch.Tensor,  # [V] logits for the first token
    beam_width: int,
    num_return: int,
    eos_ids: Set[int],
    length_penalty: float,
    early_stopping,
    max_steps: int,
) -> List[Tuple[float, List[int]]]:
    W = beam_width
    V = init_logits.shape[0]

    def logprobs_for(last: Optional[int]) -> torch.Tensor:
        logits = init_logits if last is None else transition_logits[last]
        return torch.log_softmax(logits.double(), dim=-1)

    beams: List[Tuple[float, List[int]]] = [(0.0, [])]  # single starting beam
    hyps: List[Tuple[float, List[int]]] = []
    worst = 1e9

    def hyp_add(tokens: List[int], sum_lp: float) -> None:
        nonlocal worst
        score = sum_lp / (len(tokens) ** length_penalty)
        if len(hyps) < W or score > worst:
            hyps.append((score, list(tokens)))
            if len(hyps) > W:
                hyps.sort(key=lambda x: x[0])
                hyps.pop(0)
                worst = hyps[0][0]
            else:
                worst = min(worst, score)

    def is_done(best_sum: float, cur_len: int) -> bool:
        if len(hyps) < W:
            return False
        if early_stopping is True:
            return True
        if early_stopping == "never":
            ref_len = max_steps if length_penalty > 0 else cur_len
            ha = best_sum / (ref_len ** length_penalty)
        else:
            ha = best_sum / (cur_len ** length_penalty)
        return worst >= ha

    done = False
    cur_len = 0
    for _ in range(max_steps):
        if done:
            break
        cur_len += 1
        cands: List[Tuple[float, int, int]] = []  # (score, token, beam_idx)
        for bi, (bscore, btoks) in enumerate(beams):
            lp = logprobs_for(btoks[-1] if btoks else None)
            for tok in range(V):
                cands.append((bscore + float(lp[tok]), tok, bi))
        cands.sort(key=lambda x: -x[0])
        n_sel = min(2 * W, len(cands))
        best_running = cands[0][0]
        next_beams: List[Tuple[float, List[int]]] = []
        for rank in range(n_sel):
            sc, tok, bi = cands[rank]
            if tok in eos_ids:
                if rank >= W:
                    continue
                hyp_add(beams[bi][1] + [tok], sc)
            else:
                next_beams.append((sc, beams[bi][1] + [tok]))
            if len(next_beams) == W:
                break
        beams = next_beams
        if is_done(best_running, cur_len):
            done = True

    if not done:
        for bscore, btoks in beams:
            hyp_add(btoks, bscore)
    hyps.sort(key=lambda x: -x[0])
    return hyps[:num_return]


def run_manager(
    transition_logits: torch.Tensor,
    init_logits: torch.Tensor,
    beam_width: int,
    num_return: int,
    eos_ids,
    length_penalty: float,
    early_stopping,
    max_steps: int,
) -> List[Tuple[float, List[int]]]:
    """Drive a BeamGroup with the same Markov model as the reference."""
    group = BeamGroup(
        rid="t",
        beam_width=beam_width,
        num_return=num_return,
        eos_token_ids=eos_ids,
        length_penalty=length_penalty,
        early_stopping=early_stopping,
        max_new_tokens=max_steps,
        device="cpu",
        score_dtype=torch.float64,
    )
    W = beam_width
    for _ in range(max_steps):
        if group.is_finished():
            break
        rows = []
        for i in range(W):
            toks = group.beam_tokens[i]
            last = toks[-1] if toks else None
            rows.append(init_logits if last is None else transition_logits[last])
        logits = torch.stack(rows).double()  # [W, V]
        topk_tokens, topk_logprobs = beam_sample_topk(logits, group.num_candidates)
        group.step(topk_tokens, topk_logprobs)
    return group.finalize()


class TestSamplingParamsBeam(unittest.TestCase):
    def test_defaults_unchanged(self):
        sp = SamplingParams()
        self.assertEqual(sp.num_beams, 1)
        self.assertEqual(sp.length_penalty, 1.0)
        self.assertEqual(sp.early_stopping, False)
        sp.verify(vocab_size=1000)  # must not raise

    def test_temperature_zero_greedy_without_beams(self):
        sp = SamplingParams(temperature=0.0)
        # plain greedy: top_k collapses to 1
        self.assertEqual(sp.top_k, 1)

    def test_temperature_zero_with_beams_keeps_full_vocab(self):
        sp = SamplingParams(temperature=0.0, num_beams=4, n=1)
        # beam search must keep the whole vocabulary for the top-W expansion
        self.assertEqual(sp.top_k, TOP_K_ALL)
        self.assertEqual(sp.temperature, 1.0)
        sp.verify(vocab_size=1000)

    def test_beam_rejects_nucleus_filters(self):
        for kwargs in (
            {"num_beams": 4, "top_p": 0.9},
            {"num_beams": 4, "top_k": 50},
            {"num_beams": 4, "min_p": 0.1},
        ):
            sp = SamplingParams(n=1, **kwargs)
            with self.assertRaises(ValueError):
                sp.verify(vocab_size=1000)

    def test_beam_n_bounds(self):
        with self.assertRaises(ValueError):
            SamplingParams(num_beams=2, n=5).verify(vocab_size=1000)
        SamplingParams(num_beams=4, n=4).verify(vocab_size=1000)  # ok

    def test_beam_rejects_grammar(self):
        sp = SamplingParams(num_beams=4, n=1, regex="[0-9]+")
        with self.assertRaises(ValueError):
            sp.verify(vocab_size=1000)

    def test_early_stopping_validation(self):
        SamplingParams(num_beams=4, n=1, early_stopping="never").verify(vocab_size=1000)
        with self.assertRaises(ValueError):
            SamplingParams(num_beams=4, n=1, early_stopping="sometimes").verify(
                vocab_size=1000
            )


class TestBeamSampleTopk(unittest.TestCase):
    def test_shapes_and_values(self):
        rows, vocab, W = 6, 32, 4
        logits = torch.randn(rows, vocab).double()
        toks, lps = beam_sample_topk(logits, num_candidates=2 * W)
        self.assertEqual(toks.shape, (rows, 2 * W))
        self.assertEqual(lps.shape, (rows, 2 * W))
        # descending per row
        self.assertTrue(torch.all(lps[:, :-1] >= lps[:, 1:] - 1e-9))
        # values match a manual log_softmax + topk
        ref_lp = torch.log_softmax(logits, dim=-1)
        exp_lps, exp_toks = torch.topk(ref_lp, 2 * W, dim=-1)
        torch.testing.assert_close(lps, exp_lps)
        self.assertTrue(torch.equal(toks, exp_toks))

    def test_temperature_scaling(self):
        logits = torch.randn(3, 16).double()
        t = torch.full((3, 1), 2.0, dtype=torch.float64)
        _, lps = beam_sample_topk(logits, 4, temperatures=t)
        ref = torch.log_softmax(logits / 2.0, dim=-1)
        exp, _ = torch.topk(ref, 4, dim=-1)
        torch.testing.assert_close(lps, exp)


class TestCascadeAttention(unittest.TestCase):
    def _check(self, N, H, D, Sp, Ss):
        torch.manual_seed(N * 100 + H * 10 + D)
        scale = 1.0 / math.sqrt(D)
        q = torch.randn(N, H, D, dtype=torch.float64)
        kp = torch.randn(N, H, Sp, D, dtype=torch.float64)
        vp = torch.randn(N, H, Sp, D, dtype=torch.float64)
        ks = torch.randn(N, H, Ss, D, dtype=torch.float64)
        vs = torch.randn(N, H, Ss, D, dtype=torch.float64)

        # Ground truth: attention over the union of prefix + suffix keys.
        full_out, full_lse = sdpa_with_lse(
            q, torch.cat([kp, ks], dim=2), torch.cat([vp, vs], dim=2), scale
        )
        # Cascade: prefix once + suffix, merged.
        casc_out, casc_lse = cascade_attend(
            prefix_attn=lambda: sdpa_with_lse(q, kp, vp, scale),
            suffix_attn=lambda: sdpa_with_lse(q, ks, vs, scale),
            merge_fn=merge_attn_states,
        )
        torch.testing.assert_close(casc_out, full_out, atol=1e-9, rtol=1e-7)
        torch.testing.assert_close(casc_lse, full_lse, atol=1e-9, rtol=1e-7)

    def test_cascade_matches_full_attention(self):
        for N, H, D, Sp, Ss in [
            (1, 1, 8, 5, 3),
            (4, 2, 16, 17, 2),  # beam-like: 4 queries share a longer prefix
            (8, 4, 32, 64, 1),
        ]:
            with self.subTest(N=N, H=H, D=D, Sp=Sp, Ss=Ss):
                self._check(N, H, D, Sp, Ss)

    def test_merge_is_order_invariant(self):
        # merging (A,B) and (B,A) must give the same result
        torch.manual_seed(7)
        oa = torch.randn(3, 2, 8, dtype=torch.float64)
        ob = torch.randn(3, 2, 8, dtype=torch.float64)
        la = torch.randn(3, 2, dtype=torch.float64)
        lb = torch.randn(3, 2, dtype=torch.float64)
        o1, l1 = merge_attn_states(oa, la, ob, lb)
        o2, l2 = merge_attn_states(ob, lb, oa, la)
        torch.testing.assert_close(o1, o2)
        torch.testing.assert_close(l1, l2)


class TestBeamGroupVsReference(unittest.TestCase):
    def _compare(self, vocab, W, num_return, eos_ids, lp, early, max_steps, seed):
        torch.manual_seed(seed)
        # Continuous random logits => no score ties (probability zero).
        transition = torch.randn(vocab, vocab, dtype=torch.float64)
        init = torch.randn(vocab, dtype=torch.float64)

        ref = reference_beam_search(
            transition, init, W, num_return, set(eos_ids), lp, early, max_steps
        )
        got = run_manager(
            transition, init, W, num_return, eos_ids, lp, early, max_steps
        )

        self.assertEqual(len(got), len(ref), f"count mismatch seed={seed}")
        for (gs, gt), (rs, rt) in zip(got, ref):
            self.assertEqual(gt, rt, f"tokens mismatch seed={seed}: {gt} vs {rt}")
            self.assertAlmostEqual(gs, rs, places=6, msg=f"score mismatch seed={seed}")

    def test_no_eos_various(self):
        # vocab must be >= 2*W so the sampler can return 2W candidates per beam.
        for W in (2, 3, 4):
            for seed in range(8):
                with self.subTest(W=W, seed=seed):
                    self._compare(
                        vocab=4 * W,
                        W=W,
                        num_return=W,
                        eos_ids=[],
                        lp=1.0,
                        early=False,
                        max_steps=6,
                        seed=seed,
                    )

    def test_with_eos(self):
        for seed in range(10):
            with self.subTest(seed=seed):
                self._compare(
                    vocab=12,
                    W=3,
                    num_return=2,
                    eos_ids=[0, 1],  # two EOS tokens
                    lp=1.0,
                    early=False,
                    max_steps=8,
                    seed=seed,
                )

    def test_length_penalty(self):
        for lp in (0.0, 0.5, 1.5, 2.0):
            for seed in range(5):
                with self.subTest(lp=lp, seed=seed):
                    self._compare(
                        vocab=12,
                        W=3,
                        num_return=3,
                        eos_ids=[0],
                        lp=lp,
                        early=False,
                        max_steps=8,
                        seed=seed,
                    )

    def test_early_stopping_modes(self):
        for early in (True, "never"):
            for seed in range(5):
                with self.subTest(early=early, seed=seed):
                    self._compare(
                        vocab=12,
                        W=3,
                        num_return=2,
                        eos_ids=[0],
                        lp=1.0,
                        early=early,
                        max_steps=10,
                        seed=seed,
                    )


class TestBeamGroupMechanics(unittest.TestCase):
    def test_fanout_first_step(self):
        # First step expands the single prefill state into W distinct beams, all
        # pointing back to parent beam 0.
        W = 4
        group = BeamGroup(
            rid="t",
            beam_width=W,
            num_return=W,
            eos_token_ids=[],
            max_new_tokens=4,
            score_dtype=torch.float64,
        )
        vocab = 16
        logits = torch.randn(vocab, dtype=torch.float64)
        rows = torch.stack([logits] * W)  # identical rows; -inf scores mask 1..W-1
        toks, lps = beam_sample_topk(rows, group.num_candidates)
        out = group.step(toks, lps)
        self.assertTrue(torch.all(out.parent_ptr == 0), "fan-out must point to beam 0")
        self.assertEqual(len(set(out.next_tokens.tolist())), W, "W distinct beams")
        # the W tokens are the top-W of the distribution
        ref = torch.topk(torch.log_softmax(logits, dim=-1), W).indices.tolist()
        self.assertEqual(sorted(out.next_tokens.tolist()), sorted(ref))

    def test_back_pointer_shape_and_range(self):
        W = 3
        group = BeamGroup(
            rid="t",
            beam_width=W,
            num_return=W,
            eos_token_ids=[],
            max_new_tokens=4,
            score_dtype=torch.float64,
        )
        vocab = 12
        # step 1 (fan-out)
        logits = torch.randn(W, vocab, dtype=torch.float64)
        out = group.step(*beam_sample_topk(logits, group.num_candidates))
        # step 2
        logits2 = torch.randn(W, vocab, dtype=torch.float64)
        out2 = group.step(*beam_sample_topk(logits2, group.num_candidates))
        self.assertEqual(out2.parent_ptr.shape, (W,))
        self.assertTrue(int(out2.parent_ptr.min()) >= 0)
        self.assertTrue(int(out2.parent_ptr.max()) < W)

    def test_manager_lifecycle(self):
        mgr = BeamSearchManager()
        self.assertNotIn("r1", mgr)
        mgr.add("r1", beam_width=2, num_return=1, eos_token_ids=[], max_new_tokens=3)
        self.assertIn("r1", mgr)
        vocab = 8
        out = mgr.step(
            "r1", *beam_sample_topk(torch.randn(2, vocab).double(), 4)
        )
        self.assertEqual(out.next_tokens.shape, (2,))
        mgr.remove("r1")
        self.assertNotIn("r1", mgr)


class TestBeamRowReorder(unittest.TestCase):
    def test_compute_reorder_indices(self):
        # group0 parents [0,0,1], group1 parents [2,1,1], W=3
        parent_ptr = torch.tensor([[0, 0, 1], [2, 1, 1]], dtype=torch.int64)
        plan = compute_beam_row_reorder(parent_ptr, beam_width=3)
        # global src rows: g0 offset 0 -> [0,0,1]; g1 offset 3 -> [5,4,4]
        self.assertEqual(plan.src_rows.tolist(), [0, 0, 1, 5, 4, 4])
        # parent == self: g0 [T,F,F], g1 [F,T,F]
        self.assertEqual(plan.in_place.tolist(), [True, False, False, False, True, False])
        # g0 references {0,1} -> beam 2 orphaned (row 2); g1 references {1,2} -> beam 0 (row 3)
        self.assertEqual(plan.free_beam_rows.tolist(), [2, 3])

    def test_apply_reorder_is_hazard_free(self):
        # 2 groups x W=3 = 6 beam rows; give each a distinct req_to_token row.
        W, num_groups, ctx, hist = 3, 2, 8, 4
        pool = 6
        beam_pool_indices = torch.arange(pool, dtype=torch.int64)
        req_to_token = torch.zeros(pool, ctx, dtype=torch.int64)
        for r in range(pool):
            req_to_token[r, :hist] = torch.arange(hist) + r * 100  # recognizable
        original = req_to_token.clone()
        parent_ptr = torch.tensor([[0, 0, 1], [2, 1, 1]], dtype=torch.int64)
        plan = compute_beam_row_reorder(parent_ptr, W)
        apply_beam_row_reorder(req_to_token, beam_pool_indices, plan, hist)
        # each row i now equals the ORIGINAL content of src_rows[i] (hazard-free)
        for i in range(pool):
            src = plan.src_rows[i].item()
            self.assertTrue(
                torch.equal(req_to_token[i, :hist], original[src, :hist]),
                f"row {i} should mirror src {src}",
            )

    def test_orphan_free_slots(self):
        parent_ptr = torch.tensor([[0, 0, 1], [2, 1, 1]], dtype=torch.int64)
        plan = compute_beam_row_reorder(parent_ptr, 3)
        out_cache_loc = torch.tensor([100, 101, 102, 103, 104, 105], dtype=torch.int64)
        self.assertEqual(orphan_free_slots(plan, out_cache_loc).tolist(), [102, 103])

    def test_fanout_reorder_all_point_to_beam0(self):
        # First decode step: every survivor continues beam 0 -> rows 1..W-1 orphaned.
        parent_ptr = torch.zeros(2, 4, dtype=torch.int64)  # 2 groups, W=4
        plan = compute_beam_row_reorder(parent_ptr, 4)
        self.assertEqual(plan.src_rows.tolist(), [0, 0, 0, 0, 4, 4, 4, 4])
        self.assertEqual(plan.free_beam_rows.tolist(), [1, 2, 3, 5, 6, 7])


class TestRunBeamDecodeStep(unittest.TestCase):
    def test_first_step_matches_per_group(self):
        W, num_groups, vocab = 3, 2, 12
        mgr = BeamSearchManager()
        rids = ["g0", "g1"]
        for rid in rids:
            mgr.add(rid, beam_width=W, num_return=W, eos_token_ids=[], max_new_tokens=5)
        torch.manual_seed(3)
        logits = torch.randn(num_groups * W, vocab, dtype=torch.float64)
        topk_tokens, topk_logprobs = beam_sample_topk(logits, num_candidates=2 * W)
        res = run_beam_decode_step(mgr, rids, topk_tokens, topk_logprobs, W)
        # fan-out: every beam continues beam 0
        self.assertEqual(res.parent_ptr.tolist(), [[0, 0, 0], [0, 0, 0]])
        self.assertEqual(res.reorder_plan.src_rows.tolist(), [0, 0, 0, 3, 3, 3])
        # per group, next tokens are the top-W of that group's row-0 distribution
        for g in range(num_groups):
            ref = torch.topk(torch.log_softmax(logits[g * W], -1), W).indices.tolist()
            got = res.next_tokens[g * W : (g + 1) * W].tolist()
            self.assertEqual(sorted(got), sorted(ref))
        self.assertEqual(res.next_tokens.shape, (num_groups * W,))
        self.assertEqual(list(res.finished), [False, False])

    def test_second_step_backpointers_in_range(self):
        W, vocab = 4, 16
        mgr = BeamSearchManager()
        mgr.add("g", beam_width=W, num_return=W, eos_token_ids=[], max_new_tokens=5)
        torch.manual_seed(5)
        for step in range(2):
            logits = torch.randn(W, vocab, dtype=torch.float64)
            tk, tl = beam_sample_topk(logits, 2 * W)
            res = run_beam_decode_step(mgr, ["g"], tk, tl, W)
        self.assertEqual(res.parent_ptr.shape, (1, W))
        self.assertTrue(0 <= int(res.parent_ptr.min()))
        self.assertTrue(int(res.parent_ptr.max()) < W)


if __name__ == "__main__":
    unittest.main()
