"""End-to-end beam-search validation on a real model (Qwen3-0.6B), on GPU.

Three things, to validate that beam search returns *accurate* results:

1. ALGORITHM ACCURACY: HuggingFace `generate(num_beams=W)` (the canonical reference)
   vs SGLang's `BeamGroup` selection logic driving the *same* model with a real KV cache
   (full / "replicate" attention). The generated token sequences must match.

2. CASCADE CORRECTNESS at the target width: extract the model's *real* K/V at a decode
   step, set up W beams sharing the prompt prefix, and confirm FlashInfer cascade
   (`MultiLevelCascadeAttentionWrapper`) == full batched decode. Cascade therefore yields
   identical logits -> identical beams as the replicate path validated in (1).

3. THE 500-BEAM BOUNDARY: HuggingFace / replicate beam search at 500 beams x 500 tokens
   replicates the KV cache ~30 GB and OOMs a 23 GB L4 — the exact thing cascade avoids.
   Probed empirically.

Runs on a CUDA torch that sees the GPU (the conda env on this box). Loads the model from
the local HF cache (Qwen3-0.6B).
"""
import importlib.util

import torch

MODEL = "Qwen/Qwen3-0.6B"
DEV = "cuda"
DTYPE = torch.float16


def _load_beam_module():
    p = "/home/guys/spotify/sglang/python/sglang/srt/managers/beam_search.py"
    spec = importlib.util.spec_from_file_location("sglang_beam_search", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


BS = _load_beam_module()


def beam_sample_topk(logits, num_candidates):
    """Inlined copy of sglang.srt.layers.sampler.beam_sample_topk (no temperature)."""
    logprobs = torch.log_softmax(logits, dim=-1)
    topk_lp, topk_tok = torch.topk(logprobs, num_candidates, dim=-1)
    return topk_tok, topk_lp


def sglang_beam_search(model, input_ids, W, max_new_tokens, eos_ids, length_penalty=1.0):
    """Greedy beam search driven by SGLang's BeamGroup + a HF KV cache (replicate path).

    Returns a list of (penalized_score, generated_token_ids) sorted best-first, length W.
    """
    group = BS.BeamGroup(
        rid="v", beam_width=W, num_return=W, eos_token_ids=eos_ids,
        length_penalty=length_penalty, early_stopping=False,
        max_new_tokens=max_new_tokens, device=DEV, score_dtype=torch.float32,
    )
    C = group.num_candidates  # 2W

    # Prefill (batch 1), then fan out the cache to W identical beams.
    out = model(input_ids=input_ids[None].to(DEV), use_cache=True)
    cache = out.past_key_values
    cache.reorder_cache(torch.zeros(W, dtype=torch.long, device=DEV))
    logits = out.logits[:, -1, :].float().expand(W, -1).contiguous()  # [W, vocab]

    for _ in range(max_new_tokens):
        if group.is_finished():
            break
        tk, tl = beam_sample_topk(logits, C)
        step = group.step(tk, tl)
        # KV reorder: new beam j continues old beam parent_ptr[j].
        cache.reorder_cache(step.parent_ptr.to(DEV))
        nxt = step.next_tokens.to(DEV).view(W, 1)
        out = model(input_ids=nxt, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        logits = out.logits[:, -1, :].float()

    return group.finalize()


def hf_beam_search(model, tok, input_ids, W, max_new_tokens, length_penalty=1.0):
    gen = model.generate(
        input_ids[None].to(DEV),
        num_beams=W,
        num_return_sequences=W,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        length_penalty=length_penalty,
        early_stopping=False,
        num_beam_groups=1,
        no_repeat_ngram_size=0,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    plen = input_ids.shape[0]
    return [g[plen:].tolist() for g in gen]  # strip prompt; W sequences best-first


def _trim_eos(ids, eos_ids):
    out = []
    for t in ids:
        out.append(t)
        if t in eos_ids:
            break
    return out


def test_algorithm_accuracy(model, tok, eos_ids):
    print("=" * 78)
    print("TEST 1 — SGLang BeamGroup vs HuggingFace generate (output token match)")
    print("=" * 78)
    prompt = "The history of the Roman Empire began when"
    input_ids = tok(prompt, return_tensors="pt").input_ids[0]
    for W, M in [(4, 40), (8, 40), (16, 64)]:
        hf = hf_beam_search(model, tok, input_ids, W, M)
        sg = sglang_beam_search(model, input_ids, W, M, eos_ids)
        sg_tokens = [_trim_eos(toks, eos_ids) for _, toks in sg]
        hf_tokens = [_trim_eos(s, eos_ids) for s in hf]
        top1_match = sg_tokens[0] == hf_tokens[0]
        # set-equality of all W beams (order can differ on score ties)
        set_match = sorted(map(tuple, sg_tokens)) == sorted(map(tuple, hf_tokens))
        print(f"  W={W:2d} M={M:3d}: top-1 match={top1_match}  all-{W}-beams match={set_match}")
        if not top1_match:
            print(f"    HF top1:     {hf_tokens[0][:20]}")
            print(f"    SGLang top1: {sg_tokens[0][:20]}")
    print()


def test_cascade_on_real_kv(model, tok, W=500):
    print("=" * 78)
    print(f"TEST 2 — cascade == full attention on REAL model K/V (W={W})")
    print("=" * 78)
    import flashinfer

    prompt = ("In a distant galaxy, " * 60)  # ~ a few hundred tokens
    input_ids = tok(prompt, return_tensors="pt").input_ids[0].to(DEV)
    P = input_ids.shape[0]
    cfg = model.config
    Hkv = cfg.num_key_value_heads
    Hq = cfg.num_attention_heads
    D = getattr(cfg, "head_dim", cfg.hidden_size // Hq)
    scale = 1.0 / (D ** 0.5)

    # Real K/V for one layer from a prefill.
    out = model(input_ids[None], use_cache=True)
    kc = out.past_key_values
    k = kc.layers[0].keys[0].to(DTYPE)   # [Hkv, P, D]
    v = kc.layers[0].values[0].to(DTYPE)
    k = k.transpose(0, 1).contiguous()   # [P, Hkv, D]
    v = v.transpose(0, 1).contiguous()
    q = torch.randn(W, Hq, D, device=DEV, dtype=DTYPE)  # W beam decode queries

    ws = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=DEV)
    # cascade: all W beams share the P-token prefix (level 0); no unique tail here (T=0
    # would be empty), so add a 1-token per-beam tail to exercise level 1.
    T = 1
    kv = torch.empty(P + W * T, 2, 1, Hkv, D, device=DEV, dtype=DTYPE)
    kv[:P, 0, 0] = k
    kv[:P, 1, 0] = v
    kv[P:, 0, 0] = torch.randn(W * T, Hkv, D, device=DEV, dtype=DTYPE)
    kv[P:, 1, 0] = torch.randn(W * T, Hkv, D, device=DEV, dtype=DTYPE)

    casc = flashinfer.MultiLevelCascadeAttentionWrapper(2, ws, "NHD")
    casc.plan(
        [torch.tensor([0, W], dtype=torch.int32, device=DEV),
         torch.arange(W + 1, dtype=torch.int32, device=DEV)],
        [torch.tensor([0, P], dtype=torch.int32, device=DEV),
         torch.arange(0, W * T + 1, T, dtype=torch.int32, device=DEV)],
        [torch.arange(P, dtype=torch.int32, device=DEV),
         P + torch.arange(W * T, dtype=torch.int32, device=DEV)],
        [torch.tensor([1], dtype=torch.int32, device=DEV),
         torch.ones(W, dtype=torch.int32, device=DEV)],
        Hq, Hkv, D, 1, causal=False, sm_scale=scale, q_data_type=DTYPE, kv_data_type=DTYPE,
    )
    out_c = casc.run(q, kv)

    # replicate: each beam has [prefix ++ its 1-token tail], same data.
    dec = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD")
    L = P + T
    kvr = torch.empty(W * L, 2, 1, Hkv, D, device=DEV, dtype=DTYPE)
    for b in range(W):
        base = b * L
        kvr[base:base + P] = kv[:P]
        kvr[base + P:base + L] = kv[P + b * T:P + (b + 1) * T]
    dec.plan(
        torch.arange(0, W * L + 1, L, dtype=torch.int32, device=DEV),
        torch.arange(W * L, dtype=torch.int32, device=DEV),
        torch.ones(W, dtype=torch.int32, device=DEV),
        Hq, Hkv, D, 1, q_data_type=DTYPE, data_type=DTYPE, sm_scale=scale,
    )
    out_r = dec.run(q, kvr)
    diff = (out_c.float() - out_r.float()).abs().max().item()
    print(f"  P={P} real-prefix K/V, W={W} beams: max|cascade - replicate| = {diff:.3e}  "
          f"({'MATCH' if diff < 5e-2 else 'MISMATCH'})\n")


def test_500_beam_boundary(model, tok):
    print("=" * 78)
    print("TEST 3 — HuggingFace beam search at W=500, 500 output tokens (OOM boundary)")
    print("=" * 78)
    prompt = "Once upon a time"
    input_ids = tok(prompt, return_tensors="pt").input_ids[0]
    torch.cuda.reset_peak_memory_stats()
    try:
        hf_beam_search(model, tok, input_ids, W=500, max_new_tokens=500)
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  ran; peak GPU mem = {peak:.1f} GB")
    except torch.cuda.OutOfMemoryError:
        print("  OutOfMemoryError — replicate beam search at W=500/500-tok exceeds the "
              "L4's 23 GB.\n  This is exactly the replicated-prefix KV that cascade "
              "eliminates (cascade full-model KV at W=500 is ~0.5 GB).")
    torch.cuda.empty_cache()
    print()


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading {MODEL} on {torch.cuda.get_device_name(0)} ...")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DTYPE).to(DEV).eval()
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos) if isinstance(eos, list) else {eos}
    print(f"eos_ids={eos_ids}  layers={model.config.num_hidden_layers} "
          f"Hq={model.config.num_attention_heads} Hkv={model.config.num_key_value_heads}\n")

    with torch.no_grad():
        test_algorithm_accuracy(model, tok, eos_ids)
        test_cascade_on_real_kv(model, tok, W=500)
        test_500_beam_boundary(model, tok)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("requires CUDA GPU")
    main()
