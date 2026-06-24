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

## Real fused-kernel results (`profile_cascade_flashinfer.py`)

The microbenchmark above uses a naive same-kernel attention, so it only shows the KV-memory
win. To measure the **latency** win you need the production fused kernel. This script runs
FlashInfer's actual `MultiLevelCascadeAttentionWrapper` (cascade) vs
`BatchDecodeWithPagedKVCacheWrapper` (replicate) — the kernels SGLang itself uses — at
`lpm-benchmark`'s beam-search workload shapes (`profiles/qwen3_17b_beam_search.yaml`:
prompt ~2000, beam width 15/30, short output) with Qwen3-1.7B attention dims.

```
python benchmark/beam_search_cascade/profile_cascade_flashinfer.py
```

### NVIDIA L4, Qwen3-1.7B dims (Hq=16, Hkv=8, D=128), fp16, page_size=1

Per attention call (one layer); cascade == replicate verified to 5e-4.

| workload | cascade | replicate | **latency speedup** | KV mem (per layer) | mem saving |
|---|--:|--:|--:|--:|--:|
| `bw1_6k`  P=6031 W=1  T=60 | 0.154 ms | 0.045 ms | **0.29× (slower)** | 24.9 / 24.9 MB | 1.0× |
| `bw15_2k` P=2000 W=15 T=5  | 0.152 ms | 0.500 ms | **3.30×** | 8.5 / 123 MB | 14.5× |
| `bw30_2k` P=2000 W=30 T=5  | 0.153 ms | 0.988 ms | **6.45×** | 8.8 / 246 MB | 28× |
| P=2000 W=30 T=60 | 0.153 ms | 1.017 ms | 6.66× | 15.6 / 253 MB | 16× |
| P=4000 W=30 T=5  | 0.151 ms | 1.959 ms | **13.0×** | 17 / 492 MB | 29× |
| P=8000 W=30 T=5  | 0.151 ms | 3.898 ms | **25.9×** | 33 / 984 MB | 29.5× |

**Findings**
- **Cascade latency is ~flat (~0.15 ms)** across beam width, tail length, and prefix length —
  the shared prefix KV is read *once*. Replicate scales with `W·(P+T)`, so the cascade win
  *grows* with beam width and prompt length (6.5× at W=30/P=2000 → 26× at P=8000).
- **Gate cascade on `beam_width > 1`.** At W=1 cascade is **3.4× slower** (two-pass + LSE
  merge overhead with nothing to share). SGLang's existing `use_cascade_attn` predicate (and
  the beam hook added here) already require width > 1 — keep that.
- **KV memory** drops ~`W×` for the prefix (stored once): at W=30/P=2000 that's 246 MB → 8.8 MB
  *per layer* (×28 layers: ~6.9 GB → ~246 MB per step). This is what makes wide beams feasible
  without OOM, independent of the latency win.
- **Kernel breakdown** (torch profiler): cascade = two `BatchPrefillWithPagedKVCacheKernel`
  (level-0 shared prefix + level-1 per-beam tail) + `PersistentVariableLengthMergeStates`
  (the LSE merge); replicate = one `BatchDecodeWithPagedKVCacheKernel` that is 99.5% of the
  time (980 µs at W=30). The cascade path is exactly the two-pass + `merge_state` that the
  SGLang FA3/FlashInfer backends already implement for spec-decode `topk>1` and that the beam
  hook reuses.

### Why this can't be run as a full `lpm-benchmark` here
`lpm-benchmark` drives `lpm-loader` against a *running* server (`engines/sglang/run_benchmark.sh`
→ `python -m sglang.launch_server` + `lpm-loader suite localhost:8000 --backend http`). That
needs (1) SGLang running on the GPU — blocked by the CUDA stack below; (2) beam search wired
end-to-end in SGLang — the scheduler glue in `INTEGRATION.md` is not yet landed; and (3)
`lpm-loader` from Spotify Artifactory. This script profiles the performance-critical kernel
directly at the same workload shapes, which is what determines the end-to-end beam-search win.

## Environment note

The pinned sglang 0.5.12 native stack (`torch 2.11.0+cu130`, `sgl-kernel 0.3.21+cu130`,
`flashinfer 0.6.11`) cannot run on this host: the L4 driver (550.90.07) supports only
CUDA 12.4, while the stack is built for CUDA 13. `sgl-kernel 0.3.21` ships **only**
`+cu130` wheels (verified on PyPI and the sgl-project wheel index), and the local toolkit
is nvcc 12.4 — so an end-to-end sglang GPU run here needs a from-source build of
`sgl-kernel` for cu128. These standalone scripts sidestep that by running the pure-torch
beam/cascade logic on a cu128 torch that does see the GPU.
