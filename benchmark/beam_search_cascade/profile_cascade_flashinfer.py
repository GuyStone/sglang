"""Profile the real fused cascade attention kernel vs the replicate baseline, on GPU,
at LPM-benchmark's beam-search workload shapes.

This measures the *attention* cost of one beam-search decode step (the dominant,
beam-width-sensitive part), using the actual FlashInfer kernels SGLang would use:

* CASCADE  = ``MultiLevelCascadeAttentionWrapper`` (2 levels): the W beams of a request
  attend their **shared** prompt prefix once (level 0) plus each beam's short **unique
  tail** (level 1), merged internally via log-sum-exp. KV stored once for the prefix.
* REPLICATE = ``BatchDecodeWithPagedKVCacheWrapper``: W independent sequences, each with
  its own full ``[prefix + tail]`` KV (prefix duplicated W times) — what naive beam
  search / parallel sampling does today.

Both produce the same result (validated). The comparison isolates the access pattern with
production kernels, so the latency column reflects the *real* cascade win (unlike a naive
two-pass, which only saves memory).

Workload shapes mirror ``lpm-benchmark/profiles/qwen3_17b_beam_search.yaml``
(prompt ~2000, beam width 15/30, short output) with Qwen3-1.7B attention dims. Reports
per-attention-call latency (one layer; multiply by num layers for a full step) and the
per-step KV-cache footprint.

Run on a CUDA torch that sees the GPU (e.g. the conda env on this box):
    python benchmark/beam_search_cascade/profile_cascade_flashinfer.py
"""
import torch

import flashinfer

DEV = "cuda"
DTYPE = torch.float16
# Qwen3-1.7B attention dims (GQA).
HQ, HKV, HEAD_DIM, NLAYERS = 16, 8, 128, 28
PAGE = 1  # SGLang default page size
SCALE = 1.0 / (HEAD_DIM ** 0.5)
WS = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=DEV)


def plan_cascade(P, W, T):
    casc = flashinfer.MultiLevelCascadeAttentionWrapper(2, WS, "NHD")
    qo_l0 = torch.tensor([0, W], dtype=torch.int32, device=DEV)
    kvp_l0 = torch.tensor([0, P], dtype=torch.int32, device=DEV)
    kvi_l0 = torch.arange(P, dtype=torch.int32, device=DEV)
    last_l0 = torch.tensor([1], dtype=torch.int32, device=DEV)
    qo_l1 = torch.arange(W + 1, dtype=torch.int32, device=DEV)
    kvp_l1 = torch.arange(0, W * T + 1, T, dtype=torch.int32, device=DEV)
    kvi_l1 = P + torch.arange(W * T, dtype=torch.int32, device=DEV)
    last_l1 = torch.ones(W, dtype=torch.int32, device=DEV)
    casc.plan(
        [qo_l0, qo_l1], [kvp_l0, kvp_l1], [kvi_l0, kvi_l1], [last_l0, last_l1],
        HQ, HKV, HEAD_DIM, PAGE, causal=False, sm_scale=SCALE,
        q_data_type=DTYPE, kv_data_type=DTYPE,
    )
    n_pages = P + W * T
    kv = torch.randn(n_pages, 2, PAGE, HKV, HEAD_DIM, device=DEV, dtype=DTYPE)
    return casc, kv, n_pages


def plan_replicate(P, W, T):
    dec = flashinfer.BatchDecodeWithPagedKVCacheWrapper(WS, "NHD")
    L = P + T
    indptr = torch.arange(0, W * L + 1, L, dtype=torch.int32, device=DEV)
    indices = torch.arange(W * L, dtype=torch.int32, device=DEV)
    last = torch.ones(W, dtype=torch.int32, device=DEV)
    dec.plan(
        indptr, indices, last, HQ, HKV, HEAD_DIM, PAGE,
        q_data_type=DTYPE, data_type=DTYPE, sm_scale=SCALE,
    )
    n_pages = W * L
    kv = torch.randn(n_pages, 2, PAGE, HKV, HEAD_DIM, device=DEV, dtype=DTYPE)
    return dec, kv, n_pages


def time_run(wrapper, q, kv, iters=100, warmup=20):
    for _ in range(warmup):
        wrapper.run(q, kv)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        wrapper.run(q, kv)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def validate():
    # cascade == replicate when beam b's full seq = shared prefix ++ tail_b
    torch.manual_seed(0)
    P, W, T = 64, 4, 3
    q = torch.randn(W, HQ, HEAD_DIM, device=DEV, dtype=DTYPE)
    casc, kv_c, _ = plan_cascade(P, W, T)
    rep, kv_r, _ = plan_replicate(P, W, T)
    kv_c[:P] = torch.randn(P, 2, PAGE, HKV, HEAD_DIM, device=DEV, dtype=DTYPE)
    for b in range(W):
        base = b * (P + T)
        kv_r[base : base + P] = kv_c[:P]
        kv_r[base + P : base + P + T] = kv_c[P + b * T : P + (b + 1) * T]
    d = (casc.run(q, kv_c).float() - rep.run(q, kv_r).float()).abs().max().item()
    print(f"validation: max|cascade - replicate| = {d:.3e}  "
          f"({'MATCH' if d < 5e-2 else 'MISMATCH'})\n")


def bytes_per_page():
    return 2 * PAGE * HKV * HEAD_DIM * DTYPE.itemsize


def main():
    print(f"{torch.cuda.get_device_name(0)}  Qwen3-1.7B dims (Hq={HQ},Hkv={HKV},D={HEAD_DIM}), "
          f"fp16, page_size={PAGE}\n")
    validate()

    configs = [
        ("bw1_6k_60   P=6031 W=1  T=60", 6031, 1, 60),   # LPM beam1 (no beam) sanity
        ("bw15_2k_5   P=2000 W=15 T=5", 2000, 15, 5),    # LPM
        ("bw30_2k_5   P=2000 W=30 T=5", 2000, 30, 5),    # LPM
        ("sweep T     P=2000 W=30 T=30", 2000, 30, 30),
        ("sweep T     P=2000 W=30 T=60", 2000, 30, 60),
        ("sweep P     P=4000 W=30 T=5", 4000, 30, 5),
        ("sweep P     P=8000 W=30 T=5", 8000, 30, 5),
        # Very wide beams: replicate's per-layer KV (and especially x28-layer KV)
        # explodes; cascade stays tiny. At W>=100 the full-model replicate KV alone
        # exceeds a single GPU, so cascade is what makes ultra-wide beams feasible.
        ("wide W      P=2000 W=100 T=5", 2000, 100, 5),
        ("wide W      P=2000 W=250 T=5", 2000, 250, 5),
        ("wide W      P=2000 W=500 T=5", 2000, 500, 5),
        ("wide W      P=4000 W=500 T=5", 4000, 500, 5),
    ]
    bpp = bytes_per_page()
    hdr = (f"{'config':32} | {'casc(ms)':>8} {'repl(ms)':>9} {'speedup':>7}"
           f" | {'KV casc':>8} {'KV repl':>9} {'mem x':>6} | {'repl x28L':>9}")
    print(hdr)
    print("-" * len(hdr))
    for name, P, W, T in configs:
        q = torch.randn(W, HQ, HEAD_DIM, device=DEV, dtype=DTYPE)
        casc, kv_c, pg_c = plan_cascade(P, W, T)
        t_c = time_run(casc, q, kv_c)
        mem_c = pg_c * bpp / 1e6
        try:
            rep, kv_r, pg_r = plan_replicate(P, W, T)
            t_r = time_run(rep, q, kv_r)
            mem_r = pg_r * bpp / 1e6
            full_r = pg_r * bpp * NLAYERS / 1e9
            cols = (f"{t_r:>9.4f} {t_r/t_c:>6.2f}x | {mem_c:>6.1f}M {mem_r:>8.1f}M "
                    f"{mem_r/mem_c:>5.0f}x | {full_r:>7.1f}GB")
            del rep, kv_r
        except torch.cuda.OutOfMemoryError:
            cols = f"{'OOM':>9} {'--':>7} | {mem_c:>6.1f}M {'OOM':>8} {'--':>5} | {'OOM':>9}"
            torch.cuda.empty_cache()
        print(f"{name:32} | {t_c:>8.4f} {cols}")
        del casc, kv_c
        torch.cuda.empty_cache()

    print(f"\nLatency is per attention call (one layer). A full decode step does "
          f"{NLAYERS} layers; 'repl x28L' is the full-model replicate KV cache for one "
          f"step (the L4 has 23 GB). Cascade's full-model KV stays well under 1 GB.")

    # ---- kernel breakdown (profile to understand what runs) ----
    print("\n=== torch.profiler kernel breakdown: P=2000 W=30 T=5 ===")
    P, W, T = 2000, 30, 5
    q = torch.randn(W, HQ, HEAD_DIM, device=DEV, dtype=DTYPE)
    for name, builder in [("CASCADE", plan_cascade), ("REPLICATE", plan_replicate)]:
        wrapper, kv, _ = builder(P, W, T)
        for _ in range(10):
            wrapper.run(q, kv)
        torch.cuda.synchronize()
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(20):
                wrapper.run(q, kv)
            torch.cuda.synchronize()
        print(f"\n--- {name} top CUDA kernels ---")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=4))
        del wrapper, kv
        torch.cuda.empty_cache()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("requires CUDA GPU")
    main()
