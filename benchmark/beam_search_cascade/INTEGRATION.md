# Beam search + cascade attention — integration status & remaining wiring

This tracks the SGLang-internal integration of beam search with cascade attention. It
records what has landed and gives the exact, file:line-precise spec for the remaining
scheduler/IO hooks. Line numbers are against sglang 0.5.12 (commit the branch is based on).

## What has landed (verified)

CPU-unit-tested (`test/srt/cpu/test_beam_search_cascade.py`, 24 tests):

| Piece | File |
|---|---|
| `num_beams` / `length_penalty` / `early_stopping` + validation, temp→greedy bypass | `python/sglang/srt/sampling/sampling_params.py` |
| `beam_sample_topk` + `Sampler.forward_beam` (per-row top-W) | `python/sglang/srt/layers/sampler.py` |
| `BeamGroup` / `BeamSearchManager` (HF-faithful W²→top-W, EOS pool, length penalty, early stop) + `parent_ptr` contract | `python/sglang/srt/managers/beam_search.py` |
| `cascade_attend` + `merge_attn_states` (LSE merge) | `python/sglang/srt/layers/attention/cascade.py` |
| `compute_beam_row_reorder` / `apply_beam_row_reorder` / `orphan_free_slots` / `run_beam_decode_step` + `BeamDecodeInput` carrier | `python/sglang/srt/managers/beam_search_runtime.py` |

Landed as contained, **inert** hooks (dead until a `BEAM_DECODE` batch is produced; zero
regression risk to existing paths):

| Piece | File |
|---|---|
| `ForwardMode.BEAM_DECODE` + `is_beam_decode()`; included in `is_decode()`/`is_cuda_graph()`, excluded from `is_decode_or_idle()` | `model_executor/forward_batch_info.py` |
| `self.beam_width`, generalized `use_cascade_attn` predicate, isolated `_init_beam_decode_metadata` + early-return branch in `init_forward_metadata` | `layers/attention/flashattention_backend.py` |

## KV-reorder design (improvement over the original plan)

The original plan called for copy-on-reorder of tail KV via `move_kv_cache`. During
implementation we found a cheaper, correct alternative: a **`req_to_token` row repoint**.
Because the W beams of a request have identical token *history* up to the current step
(they diverge only in the token chosen *now*, whose KV is computed on the next forward),
survivor beam `j` inheriting parent `p` is just `req_to_token[row_j][:L] =
req_to_token[row_p][:L]` — the historical KV slots are shared read-only across beams
(exactly like radix prefix sharing), and only the next token gets a fresh slot. No KV
*data* copy. `compute_beam_row_reorder` produces the gather indices; `orphan_free_slots`
identifies the just-written slots of beams that no survivor inherits (free them).

## Remaining wiring (thin glue; needs a runtime GPU env to validate)

Each item references the exact hook from the integration map.

### 1. Scheduler state + request entry
- `Scheduler.__init__`: `self.beam_search_manager = BeamSearchManager()`.
- `io_struct.py:_handle_parallel_sampling` + `tokenizer_manager.py:1424-1460`: when
  `sampling_params.num_beams > 1`, set `parallel_sample_num = 1` (single logical request;
  do NOT duplicate). Carry `num_beams` through.

### 2. Fan-out 1→W at prefill→decode transition
- `scheduler.py:get_next_batch_to_run` (~2496) / `schedule_batch.py:prepare_for_decode`
  (~2286): for a beam request, materialize W contiguous rows. Allocate W `req_pool_idx`
  rows (`ReqToTokenPool.alloc`, `memory_pool.py:160`); write the shared prompt prefix into
  all W rows identically (radix prefix sharing makes the slots shared). Register the group:
  `beam_search_manager.add(rid, beam_width, n, eos_token_ids, length_penalty, early_stopping, max_new_tokens, device)`.
- Set `batch.forward_mode = ForwardMode.BEAM_DECODE` for these batches and set
  `backend.beam_width = W`.

### 3. Force non-overlap for beam batches
- `scheduler.py:is_disable_overlap_for_batch` (~1618): return `True` if the batch has beam
  requests (the beam step + KV reorder is a hard data dependency for the next forward).

### 4. Beam sampling
- `model_runner.py:sample` (~3493): add `sample_beam(logits_output, forward_batch)` →
  `self.sampler.forward_beam(logits_output, forward_batch.sampling_info, num_candidates=2*W)`.
- `tp_worker.py:forward_batch_generation` (~505-509): for beam batches call `sample_beam`
  and stash `beam_topk_tokens [G*W, 2W]`, `beam_topk_logprobs [G*W, 2W]` on
  `GenerationBatchResult` (`managers/utils.py:26` — add two optional fields).

### 5. Beam step + KV reorder + cascade metadata
- `scheduler_output_processor_mixin.py:process_batch_result_decode` (~468), before the
  per-req append loop (line ~554): call
  `run_beam_decode_step(self.beam_search_manager, group_rids, beam_topk_tokens, beam_topk_logprobs, W)`
  → `BeamDecodeStepResult`. Then:
  - `apply_beam_row_reorder(req_to_token, beam_pool_indices, result.reorder_plan, history_len)`.
  - free `orphan_free_slots(result.reorder_plan, out_cache_loc)` via the allocator.
  - feed `result.next_tokens` as the next forward's input ids.
  - skip per-row `req.check_finished`; use `result.finished` (group-level) instead.
- Build a `BeamDecodeInput` (prefix_lens = group divergence point D_g, prefix_page_table,
  tail_lens, tail_page_table) from the beam row layout and attach to `batch.spec_info` so
  `FlashAttentionBackend._init_beam_decode_metadata` can build the cascade metadata. The
  shared prefix advances (D_g grows) whenever all W beams of a group agree on a token,
  which bounds tail length and keeps cascade beneficial ("prefix re-commit").

### 6. Output assembly (top-n hypotheses)
- `scheduler_output_processor_mixin.py:stream_output_generation` (~1024), finished block
  (~1088): for a finished beam group call `beam_search_manager.finalize(rid)` → top-n
  `(score, tokens)`; emit n choices (`CompletionResponse.choices`, `protocol.py:380`).
  Then `beam_search_manager.remove(rid)`. Requires the detokenizer path
  (`BatchTokenIDOutput.output_ids`, ~1265) to carry n sequences per request — the one
  protocol change of note.

### 7. CUDA-graph capture (optional, perf)
- `flashattention_backend.py` (~1405-1463): add beam-decode metadata buffers parallel to
  the `topk>1` ones; route on `is_beam_decode()` in capture/replay. Until then, run beam
  batches eager.

## First-cut scope (recommended)
FA3 backend, non-MLA, no sliding-window cascade (SWA layers replicate), overlap disabled,
non-streaming, no grammar/structured output, no spec-decode combination. Uniform `W` per
beam batch.

## Validation plan once a GPU stack exists
1. `num_beams=1` bit-identical to current decode.
2. W=4 greedy beams match a HuggingFace `model.generate(num_beams=4)` reference (token-level).
3. Cascade output ≈ replicate-path output (same beams) within tolerance.
4. Perf: cascade vs replicate on long-prompt/short-gen (see `validate_and_bench.py`).
