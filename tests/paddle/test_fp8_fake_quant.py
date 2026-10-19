# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""MXFP8 fake-quant precision study for FlashMask attention (paddle).

Purpose: estimate, before any FP8 kernel exists, how much error each low-precision
choice adds to O / dQ / dK / dV. Every GEMM input is quantized and immediately
dequantized ("fake quant"); the GEMM itself runs in FP32. If the quantizer below is
bit-identical to what a kernel does, the kernel can only differ by accumulation order.

Quantizer semantics follow Meta LP-FA4 (lp_fa4/mxfp8.py, flash_fwd_sm100.py):
  * MXFP8: 32 consecutive values along the GEMM contraction dim share one E8M0 scale,
    scale = 2^ceil(log2(amax / fmt_max)) ("RCEIL"); payload cast RN-satfinite.
  * P uses a constant scale 256 (E4M3 code = RN(256 * p)), no per-block amax.
  * Forward P is what the kernel actually quantizes: exp(S - m_run), where m_run is the
    running row max after visiting KV tiles in REVERSE order (FlashMask fwd does that),
    later rescaled by exp(m_run - m_final) in FP32.

Metrics (x = tested tensor, r = FP32 reference, d = x - r, all reductions in FP64),
each taken from the paper that reports it:
  cos      <x, r> / (|x|_2 |r|_2)                SageAttention 1-3, SageBwd, Attn-QAT, FP4-FA4
  relL1    sum|d| / sum|r|                       SageAttention 1-3 ("relative L1 distance")
  relL2    |d|_2 / |r|_2                         FP4-FA4 (eq. 23), SageBwd
  SQNR     10 log10(|r|^2 / |d|^2) dB            Meta LP-FA4 blog
  RMSE     sqrt(mean d^2)                        SageAttention 1-3, FP4-FA4
  maxabs   max|d|, and maxabs / maxabs(BF16 kernel)   Attn-QAT Table 7; FlashAttention test rule
  maxrel   max|d| / max|r|                       (scale-free form of maxabs; ours)
Per-element |d|/|r| is not reported: it is undefined where r ~ 0, which is why the
papers above use the normalized sums instead.
Diagnostics:
  P lost mass   probability mass whose forward FP8 code is 0 (FP4-FA4 Table 3)
  dS row sum    median_i |sum_j dS_ij| / sum_j |dS_ij|, rows i >= 64 with a nonzero
                denominator (Delta-Matching eq. 23, pre-cast); 0 in exact arithmetic

Run:  pytest -s tests/paddle/test_fp8_fake_quant.py
"""

import json
import math
import os
from dataclasses import dataclass

import paddle
import pytest

from flashmask.cute.interface import flashmask_attention
from tests.paddle.attention_ref import attention_ref
from tests.paddle.generate_startend_row_indices import (
    startend_row_indices_to_attn_bias,
    generate_none_mask,
    generate_sliding_window_mask,
    generate_causal_document_mask,
)

FMT_MAX = {"e4m3": 448.0, "e5m2": 57344.0}
FMT_DTYPE = {"e4m3": "float8_e4m3fn", "e5m2": "float8_e5m2"}
BLOCK = 32       # MXFP8 scale group
TILE_N = 128     # forward KV tile (running-max granularity)
P_SCALE = 256.0  # constant P scale (P-Cast / LP-FA4 const_p / Delta-Matching)


# ----------------------------------------------------------------------------
# Quantizers (all return dequantized FP32)
# ----------------------------------------------------------------------------
def _e8m0_rceil(ratio):
    """2^ceil(log2(ratio)), exact for powers of two (frexp: ratio = m * 2^e, m in [0.5, 1))."""
    m, e = paddle.frexp(ratio)
    e = paddle.where(m == 0.5, e - 1, e)
    return paddle.pow(paddle.full_like(e, 2.0), paddle.clip(e, min=-127.0))


def _cast(x, fmt):
    return x.astype(FMT_DTYPE[fmt]).astype("float32")


def mx_quant(x, axis, fmt="e4m3"):
    """MXFP8 fake quant, 1x32 blocks along `axis` (the contraction dim of the consuming GEMM)."""
    xt = paddle.moveaxis(x.astype("float32"), axis, -1)
    shape = xt.shape
    blk = xt.reshape(shape[:-1] + [shape[-1] // BLOCK, BLOCK])
    scale = _e8m0_rceil(blk.abs().max(-1, keepdim=True) / FMT_MAX[fmt])
    return paddle.moveaxis((_cast(blk / scale, fmt) * scale).reshape(shape), -1, axis)


def mx_quant_square(x, fmt="e4m3"):
    """32x32 blocks over the last two dims: one copy is valid for both contraction directions."""
    *lead, r, c = x.shape
    blk = x.astype("float32").reshape(lead + [r // BLOCK, BLOCK, c // BLOCK, BLOCK])
    amax = blk.abs().max(axis=[-3, -1], keepdim=True)
    scale = _e8m0_rceil(amax / FMT_MAX[fmt])
    return (_cast(blk / scale, fmt) * scale).reshape(x.shape)


def p_quant(p):
    """P in [0, 1] with the constant scale 256."""
    return _cast(p * P_SCALE, "e4m3") / P_SCALE


# ----------------------------------------------------------------------------
# Explicit FlashAttention fwd + bwd with a fake-quant switch per GEMM
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Recipe:
    qk: bool = False        # fwd S and bwd S recompute: Q, K in MXFP8 (same copy in both)
    pv: bool = False        # fwd PV and bwd dV: P with scale 256, V / dO in MXFP8
    pv_v: bool = True       # with pv: also MXFP8-quantize V (off: V is used as given, e.g. the
                            # dequantized row-wise FP8 V of the d > 256 kernel)
    dp: bool = False        # bwd dP = dO V^T in MXFP8 (else BF16 inputs)
    ds: bool = False        # bwd dQ = dS K, dK = dS^T Q in MXFP8
    do_fmt: str = "e4m3"    # payload format of dO
    delta: str = "stale"    # "stale": rowsum(dO * O_saved); "matched": rowsum(P * dP_used)
    square: bool = False    # 32x32 blocks for tensors read in two directions (Q, K, V, dO, dS)
    tau: float = 0.0        # fwd running max moves only if it grows by > tau (log2 units), as the
                            # kernel's rescale_threshold; MXFP8 kernel: log2(448 / 256)


def _running_max(tile_max, tau):
    """Row max the kernel holds at each KV tile (tiles visited last to first). With tau > 0 the
    max is only replaced when the new one is larger by more than tau in log2 units."""
    if tau == 0.0:
        return paddle.flip(paddle.cummax(paddle.flip(tile_max, [-1]), axis=-1)[0], [-1])
    n = tile_max.shape[-1]
    cur, out = tile_max[..., n - 1], [None] * n
    out[n - 1] = cur
    for j in range(n - 2, -1, -1):
        t = tile_max[..., j]
        cur = paddle.where((t - cur) * math.log2(math.e) > tau, t, cur)
        out[j] = cur
    return paddle.stack(out, -1)


def fake_attention(q, k, v, do, bias, r):
    """q, k, v, do: [b, h, s, d] FP32 holding BF16 values; k, v already expanded to h heads.
    bias: additive 0 / -inf mask broadcastable to [b, h, sq, sk], or None.
    Returns o, dq, dk, dv (per q-head) and a dict of diagnostics."""
    scale = 1.0 / math.sqrt(q.shape[-1])

    def mx(x, axis, on, fmt="e4m3"):  # x is contracted along `axis` by the consuming GEMM
        if not on:
            return x
        return mx_quant_square(x, fmt) if r.square else mx_quant(x, axis, fmt)

    # ---- forward ----
    s = paddle.matmul(mx(q, -1, r.qk), mx(k, -1, r.qk), transpose_y=True) * scale
    if bias is not None:
        s = s + bias
    m = s.max(-1, keepdim=True)
    m = paddle.where(paddle.isinf(m), paddle.zeros_like(m), m)   # fully masked rows
    l = paddle.exp(s - m).sum(-1, keepdim=True)                  # row sum from pre-cast P
    l = paddle.where(l == 0, paddle.ones_like(l), l)
    lost_mass = 0.0
    if r.pv:
        b_, h_, sq, sk = s.shape
        st = s.reshape([b_, h_, sq, sk // TILE_N, TILE_N])
        # running max seen by the kernel at each tile (KV tiles visited last-to-first)
        m_run = _running_max(st.max(-1), r.tau)
        m_run = paddle.where(paddle.isinf(m_run), m.expand_as(m_run), m_run)
        p_tile = paddle.exp(st - m_run.unsqueeze(-1))
        p_hat = p_quant(p_tile)
        alpha = paddle.exp(m_run - m).unsqueeze(-1)
        p_norm = p_tile * alpha / l.unsqueeze(-1)                 # exact normalized P
        zeroed = ((p_tile > 0) & (p_hat == 0)).astype("float32")
        lost_mass = _sum64(zeroed * p_norm) / _sum64(p_norm)
        del p_norm, zeroed
        p_fwd = (p_hat * alpha).reshape(s.shape)
    else:
        p_fwd = paddle.exp(s - m)
    o = paddle.matmul(p_fwd, mx(v, -2, r.pv and r.pv_v)) / l
    lse = m + paddle.log(l)
    o_saved = o.astype("bfloat16").astype("float32")             # O is stored in BF16

    # ---- backward ----
    p = paddle.exp(s - lse)                                      # same S as forward
    dp = paddle.matmul(mx(do, -1, r.dp, r.do_fmt), mx(v, -1, r.dp), transpose_y=True)
    p_dv = p_quant(p) if r.pv else p                             # dV reads the quantized P
    dv = paddle.matmul(p_dv, mx(do, -2, r.pv, r.do_fmt), transpose_x=True)
    if r.delta == "stale":
        delta = (do * o_saved).sum(-1, keepdim=True)
    else:
        delta = (p * dp).sum(-1, keepdim=True)                  # matches the dP actually used
    ds = p * (dp - delta)                                        # dS reads the FP32 P
    dq = paddle.matmul(mx(ds, -1, r.ds), mx(k, -2, r.ds)) * scale
    dk = paddle.matmul(mx(ds, -2, r.ds), mx(q, -2, r.ds), transpose_x=True) * scale
    stats = dict(p_lost_mass=lost_mass, ds_rowsum=_rowsum_residual(ds))
    return o, dq, dk, dv, stats


def _sum64(x):
    """FP64 sum of a large FP32 tensor, one head at a time to bound memory."""
    return sum(float(x[:, h].astype("float64").sum()) for h in range(x.shape[1]))


def _rowsum_residual(ds, skip_rows=64):
    """Delta-Matching eq. 23 (pre-cast): median over rows of |sum_j dS| / sum_j |dS|."""
    ratios = []
    for h in range(ds.shape[1]):  # per head, FP64 row sums
        x = ds[:, h, skip_rows:, :].astype("float64")
        num, den = x.sum(-1).abs(), x.abs().sum(-1)
        keep = den > 0
        ratios.append(paddle.masked_select(num, keep) / paddle.masked_select(den, keep))
    return float(paddle.median(paddle.concat(ratios)))


# ----------------------------------------------------------------------------
# Test harness
# ----------------------------------------------------------------------------
TRUTH = Recipe(delta="matched")  # no quantization, exact delta
RECIPES = {
    "qk_only": Recipe(qk=True),
    "fp8_stale": Recipe(qk=True, pv=True, dp=True, ds=True),            # LP-FA4 / FP4-FA4 delta
    "fp8_matched": Recipe(qk=True, pv=True, dp=True, ds=True, delta="matched"),
    "dp_bf16": Recipe(qk=True, pv=True, ds=True),                       # SageBwd keeps dP high
    "do_e5m2": Recipe(qk=True, pv=True, dp=True, ds=True, delta="matched", do_fmt="e5m2"),
    "square": Recipe(qk=True, pv=True, dp=True, ds=True, delta="matched", square=True),
}
MASKS = {
    "causal": lambda b, sq, sk, h: generate_none_mask(b, sq, sk, h, causal=True),
    "causal_document": generate_causal_document_mask,
    "sliding_window": generate_sliding_window_mask,
}
B, S, H, H_KV = 1, 8192, 8, 2


def metrics(x, ref):
    """Error of x against ref; definitions and sources are listed in the module docstring."""
    x, ref = x.astype("float64"), ref.astype("float64")
    d = x - ref
    d2, r2 = float((d * d).sum()), float((ref * ref).sum())
    maxabs = float(d.abs().max())
    return dict(
        cos=float((x * ref).sum()) / math.sqrt(float((x * x).sum()) * r2),
        relL1=float(d.abs().sum() / ref.abs().sum()),
        relL2=math.sqrt(d2 / r2),
        SQNR=10.0 * math.log10(r2 / d2) if d2 > 0 else float("inf"),
        RMSE=math.sqrt(d2 / d.numel()),
        maxabs=maxabs,
        maxrel=maxabs / float(ref.abs().max()),
    )


def _rel(x, ref):
    return metrics(x, ref)["relL2"]


def test_metrics_and_quantizers():
    """Hand-checkable values, so the numbers in the study can be trusted."""
    ref = paddle.to_tensor([3.0, -4.0])
    m = metrics(paddle.to_tensor([3.0, -3.0]), ref)              # d = [0, 1]
    assert abs(m["relL2"] - 1 / 5) < 1e-12 and abs(m["relL1"] - 1 / 7) < 1e-12
    assert abs(m["cos"] - 21 / (5 * math.sqrt(18))) < 1e-12
    assert abs(m["SQNR"] - 10 * math.log10(25)) < 1e-12
    assert abs(m["RMSE"] - math.sqrt(0.5)) < 1e-12 and m["maxabs"] == 1 and m["maxrel"] == 0.25
    # RCEIL: amax = 448 * 2^k maps exactly to the E4M3 max; anything above rounds the scale up.
    assert float(_e8m0_rceil(paddle.to_tensor([1.0, 1.5, 0.25]))[1]) == 2.0
    x = paddle.to_tensor([[448.0 * 4] + [1.0] * 31])
    assert float(mx_quant(x, -1)[0, 0]) == 448.0 * 4                # scale 4, code 448
    assert float(p_quant(paddle.to_tensor([1.0]))[0]) == 1.0      # 256 is exact in E4M3
    assert float(p_quant(paddle.to_tensor([2.0 ** -19]))[0]) == 0.0  # below 2^-10 / 256


def _to_bhsd(x, rep=1):
    x = x.astype("float32").transpose([0, 2, 1, 3])
    return paddle.repeat_interleave(x, rep, axis=1) if rep > 1 else x


def _to_bshd(x, groups=1):
    if groups > 1:  # sum GQA gradients back onto the kv heads
        b, h, s, d = x.shape
        x = x.reshape([b, h // groups, groups, s, d]).sum(2)
    return x.transpose([0, 2, 1, 3])


def _dense_bias(startend, causal, sq, sk):
    if startend is not None:
        return startend_row_indices_to_attn_bias(startend, sq, 1, paddle.float32, causal)
    if causal:
        return paddle.triu(paddle.full([sq, sk], float("-inf")), 1).reshape([1, 1, sq, sk])
    return None


@pytest.mark.parametrize("logit_std", [1.0, 3.0])
@pytest.mark.parametrize("d", [128, 256])
@pytest.mark.parametrize("mask", list(MASKS))
def test_fp8_fake_quant(mask, d, logit_std):
    paddle.seed(2026)
    g = H // H_KV
    q = (paddle.randn([B, S, H, d]) * logit_std).astype("bfloat16")
    k = paddle.randn([B, S, H_KV, d]).astype("bfloat16")
    v = paddle.randn([B, S, H_KV, d]).astype("bfloat16")
    do = paddle.randn([B, S, H, d]).astype("bfloat16")
    startend, causal = MASKS[mask](B, S, S, 1)
    bias = _dense_bias(startend, causal, S, S)

    def run(recipe):
        o, dq, dk, dv, st = fake_attention(
            _to_bhsd(q), _to_bhsd(k, g), _to_bhsd(v, g), _to_bhsd(do), bias, recipe
        )
        out = dict(O=_to_bshd(o), dQ=_to_bshd(dq), dK=_to_bshd(dk, g), dV=_to_bshd(dv, g))
        paddle.device.cuda.empty_cache()  # the N x N temporaries are large; avoid fragmentation
        return out, st

    truth, _ = run(TRUTH)

    # Harness self-check: the explicit FP32 fwd/bwd must reproduce autograd on attention_ref.
    qr, kr, vr = (x.astype("float32") for x in (q, k, v))
    for x in (qr, kr, vr):
        x.stop_gradient = False
    o_ref, _ = attention_ref(qr, kr, vr, causal=causal, attn_bias=bias)
    o_ref.backward(do.astype("float32"))
    for name, ref in dict(O=o_ref, dQ=qr.grad, dK=kr.grad, dV=vr.grad).items():
        assert _rel(truth[name], ref) < 1e-4, f"harness mismatch on {name}"

    # BF16 FlashMask kernel on the same inputs: the yardstick for "acceptable" error.
    qb, kb, vb = (x.detach().clone() for x in (q, k, v))
    for x in (qb, kb, vb):
        x.stop_gradient = False
    o_bf16 = flashmask_attention(qb, kb, vb, startend_row_indices=startend, causal=causal)
    o_bf16.backward(do)
    bf16 = dict(O=o_bf16, dQ=qb.grad, dK=kb.grad, dV=vb.grad)

    rows = {"bf16_kernel": ({n: metrics(t, truth[n]) for n, t in bf16.items()}, None)}
    for name, recipe in RECIPES.items():
        out, st = run(recipe)
        for n, t in out.items():
            assert bool(paddle.isfinite(t).all()), f"{name}: non-finite {n}"
        rows[name] = ({n: metrics(t, truth[n]) for n, t in out.items()}, st)
    _report(f"{mask} d={d} logit_std={logit_std}", rows)
    paddle.device.cuda.empty_cache()


def _report(title, rows):
    """Daily view: cos / relL2 / maxrel per tensor plus the two diagnostics.
    All metrics (incl. relL1, SQNR, RMSE, maxabs) go to FP8_FQ_JSON as JSON lines if set."""
    tensors = list(rows["bf16_kernel"][0])
    print(f"\n===== {title}   each cell: cos / relL2 / maxrel vs FP32")
    print(f"{'recipe':<12}" + "".join(f"{t:>27}" for t in tensors) + f"{'P lost':>10}{'dS rsum':>10}")
    for name, (m, st) in rows.items():
        cells = "".join(
            f"{m[t]['cos']:>11.6f}{m[t]['relL2']:>8.1e}{m[t]['maxrel']:>8.1e}" for t in tensors
        )
        diag = f"{st['p_lost_mass']:>10.1e}{st['ds_rowsum']:>10.1e}" if st else ""
        print(f"{name:<12}{cells}{diag}")
    path = os.environ.get("FP8_FQ_JSON")
    if path:
        with open(path, "a") as f:
            for name, (m, st) in rows.items():
                f.write(json.dumps(dict(case=title, recipe=name, metrics=m, stats=st)) + "\n")
