# Beam search + cascade attention — validation artifacts

This directory holds GPU validation / microbenchmark scripts for the beam-search-with-
cascade-attention work (control core + cascade primitive landed in
`python/sglang/srt/managers/beam_search.py`, `python/sglang/srt/layers/attention/cascade.py`,
`python/sglang/srt/layers/sampler.py:beam_sample_topk`). Unit tests:
`test/srt/cpu/test_beam_search_cascade.py`.

## `validate_and_bench.py`

Standalone (loads the pure-torch modules by path; no `sglang` package import, so it runs
on any CUDA torch). Validates cascade numerics on GPU, runs the beam loop on cuda, and
microbenchmarks the beam decode access pattern.

```
python benchmark/beam_search_cascade/validate_and_bench.py
```

## Results on NVIDIA L4 (sm_89), torch 2.9.1+cu128, fp16

Numerics: `cascade_attend(prefix, suffix) == full attention over the union` to within
fp32/fp16/bf16 tolerance for all tested shapes (incl. 16 queries sharing a 1024-token
prefix). Beam loop runs on cuda and returns valid hypotheses.

Access-pattern microbenchmark (`W` beams, `P`-token shared prefix, `T=16` tails):

| W | P | KV memory: replicate | cascade | saving |
|--:|--:|--:|--:|--:|
| 4 | 512 | 17.3 MB | 4.7 MB | 3.67× |
| 16 | 2048 | 270.5 MB | 18.9 MB | 14.33× |
| 16 | 8192 | 1075.8 MB | 69.2 MB | 15.55× |

**KV-memory saving is large and exact** — footprint `P + W·T` (cascade) vs `W·(P+T)`
(replicate). This is the headline win for beam search, where the prefix dominates.

**Latency caveat:** the script uses the *same* hand-written attention kernel for both
paths, which does **not** exploit shared-prefix reuse — so it shows cascade as *slower*
(extra two-pass + merge over identical FLOPs). The cascade **latency** win requires a
fused, bandwidth-optimized kernel that reads the shared prefix once and reuses it across
the W beams — exactly FlashInfer's `merge_state` two-pass / `sgl_kernel`'s
`merge_state_v2`, which sglang already uses for speculative decoding's `target_verify`
(`flashattention_backend.py`). The implementation must therefore route beam decode through
that fused path, not a naive two-pass.

## Environment note

The pinned sglang 0.5.12 native stack (`torch 2.11.0+cu130`, `sgl-kernel 0.3.21+cu130`,
`flashinfer 0.6.11`) cannot run on this host: the L4 driver (550.90.07) supports only
CUDA 12.4, while the stack is built for CUDA 13. `sgl-kernel 0.3.21` ships **only**
`+cu130` wheels (verified on PyPI and the sgl-project wheel index), and the local toolkit
is nvcc 12.4 — so an end-to-end sglang GPU run here needs a from-source build of
`sgl-kernel` for cu128. These standalone scripts sidestep that by running the pure-torch
beam/cascade logic on a cu128 torch that does see the GPU.
