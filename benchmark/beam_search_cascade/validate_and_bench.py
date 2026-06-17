"""GPU validation + microbenchmark for beam-search cascade attention.

Standalone: loads the pure-torch beam/cascade modules by path (no `sglang` package
import), so it runs on any CUDA-enabled torch — including environments where the full
sglang native stack (sgl-kernel / flashinfer) is not installed for the local CUDA.

Run:
    python benchmark/beam_search_cascade/validate_and_bench.py

What it does:
  1. Numerics: cascade_attend(prefix, suffix) == full attention over the union, on GPU,
     in fp32 / fp16 / bf16, for several beam-like shapes.
  2. BeamGroup executes on cuda tensors and returns valid hypotheses.
  3. Microbenchmark: cascade vs replicate for the beam decode access pattern (W beams
     sharing a P-token prefix, T-token tails), reporting latency and KV-memory footprint.

NOTE on the latency column: the *same* hand-written attention (returning LSE) is used for
both paths, so the comparison isolates the access pattern rather than kernel quality.
This naive kernel does not exploit shared-prefix reuse, so it shows only the KV-MEMORY
win, not the bandwidth/latency win. The latency win is realized by a fused cascade kernel
(FlashInfer MultiLevelCascadeAttentionWrapper / sgl_kernel merge_state_v2, which sglang
already uses for speculative decoding) that reads the shared prefix once and reuses it
across the W beams. Treat the KV-memory column as exact and the latency column as a
conservative lower bound that motivates routing beam search through the fused path.
"""
import importlib.util
import math
from pathlib import Path

import torch

REPO_PY = Path(__file__).resolve().parents[2] / "python"


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO_PY / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


casc = _load("sglang/srt/layers/attention/cascade.py", "casc")
bs = _load("sglang/srt/managers/beam_search.py", "bs")


def attn_with_lse(q, k, v, scale):
    scores = torch.einsum("nhd,nhsd->nhs", q, k).float() * scale
    lse = torch.logsumexp(scores, dim=-1)
    out = torch.einsum("nhs,nhsd->nhd", torch.softmax(scores, dim=-1).to(v.dtype), v)
    return out, lse


def check_numerics(dev):
    tols = {torch.float32: 1e-4, torch.float16: 4e-3, torch.bfloat16: 2e-2}
    cases = [(4, 8, 64, 128, 4), (8, 16, 128, 512, 2), (16, 8, 128, 1024, 8)]
    print("=== numerics: cascade_attend == full attention (GPU) ===")
    for dtype, atol in tols.items():
        for N, H, D, P, T in cases:
            torch.manual_seed(N + P)
            scale = 1.0 / math.sqrt(D)
            q = torch.randn(N, H, D, device=dev, dtype=dtype)
            kp = torch.randn(N, H, P, D, device=dev, dtype=dtype)
            vp = torch.randn(N, H, P, D, device=dev, dtype=dtype)
            ks = torch.randn(N, H, T, D, device=dev, dtype=dtype)
            vs = torch.randn(N, H, T, D, device=dev, dtype=dtype)
            fo, fl = attn_with_lse(q, torch.cat([kp, ks], 2), torch.cat([vp, vs], 2), scale)
            co, cl = casc.cascade_attend(
                lambda: attn_with_lse(q, kp, vp, scale),
                lambda: attn_with_lse(q, ks, vs, scale),
                merge_fn=casc.merge_attn_states,
            )
            torch.testing.assert_close(co.float(), fo.float(), atol=atol, rtol=atol)
            torch.testing.assert_close(cl, fl, atol=atol, rtol=atol)
    print("  all shapes/dtypes pass\n")


def check_beam(dev):
    print("=== BeamGroup on cuda ===")
    W, vocab, steps = 4, 64, 8
    g = bs.BeamGroup("g", W, W, eos_token_ids=[0], max_new_tokens=steps, device=dev)
    trans = torch.randn(vocab, vocab, device=dev)
    init = torch.randn(vocab, device=dev)
    while not g.is_finished():
        rows = [init if not g.beam_tokens[i] else trans[g.beam_tokens[i][-1]] for i in range(W)]
        lp = torch.log_softmax(torch.stack(rows), dim=-1)
        tl, tk = torch.topk(lp, g.num_candidates, dim=-1)
        g.step(tk, tl)
    res = g.finalize()
    print(f"  {len(res)} hypotheses, top score {res[0][0]:.3f}\n")


def _time_ms(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def benchmark(dev, dtype=torch.float16, H=16, D=128):
    print("=== cascade vs replicate (beam decode access pattern) ===")
    print(f"{'W':>3} {'P':>5} {'T':>3} | {'repl':>8} {'casc':>8} {'lat x':>6}"
          f" | {'KV repl':>8} {'KV casc':>8} {'KV x':>6}")
    print("-" * 70)
    bpe = torch.finfo(dtype).bits // 8
    for P in (512, 2048, 8192):
        for W in (4, 8, 16):
            T = 16
            scale = 1.0 / math.sqrt(D)
            q = torch.randn(W, H, D, device=dev, dtype=dtype)
            kf = torch.randn(W, H, P + T, D, device=dev, dtype=dtype)
            vf = torch.randn(W, H, P + T, D, device=dev, dtype=dtype)
            kp = torch.randn(1, H, P, D, device=dev, dtype=dtype).expand(W, H, P, D)
            vp = torch.randn(1, H, P, D, device=dev, dtype=dtype).expand(W, H, P, D)
            kt = torch.randn(W, H, T, D, device=dev, dtype=dtype)
            vt = torch.randn(W, H, T, D, device=dev, dtype=dtype)

            def repl():
                attn_with_lse(q, kf, vf, scale)

            def cascade():
                op, lp = attn_with_lse(q, kp, vp, scale)
                os_, ls = attn_with_lse(q, kt, vt, scale)
                casc.merge_attn_states(op, lp, os_, ls)

            tr, tc = _time_ms(repl), _time_ms(cascade)
            kvr = 2 * W * (P + T) * H * D * bpe
            kvc = 2 * (P + W * T) * H * D * bpe
            print(f"{W:>3} {P:>5} {T:>3} | {tr:>7.3f}m {tc:>7.3f}m {tr/tc:>5.2f}x"
                  f" | {kvr/1e6:>6.1f}M {kvc/1e6:>6.1f}M {kvr/kvc:>5.2f}x")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires a CUDA GPU.")
    dev = "cuda"
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}\n")
    check_numerics(dev)
    check_beam(dev)
    benchmark(dev)
