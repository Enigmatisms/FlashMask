# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""FP8 FlashMask forward kernels vs their fake-quant model (paddle, SM100): MXFP8 (d <= 256)
and row-wise FP8 (d > 256, the folded 2-CTA config).

Each kernel (flash_fwd_sm100.py, blockscaled=True / rowwise_fp8=True) must compute exactly what
the fake-quant reference in test_fp8_fake_quant.py describes, up to FP32 accumulation order and the
hardware exp2: same dequantized Q/K/V, P = E4M3(2^8 * exp(S - m_running)) per KV tile in
reverse order (running max moved only past the kernel's rescale threshold), row sum
from the pre-cast P.

Pass criterion (no hand-tuned constant): the kernel's distance to that model must be at
least 10x smaller than the model's own quantization error vs FP32, i.e. what is left is
noise, not a different algorithm. LSE uses the FP32 row sum in both, so it must match FP32.

Run:  pytest -s tests/paddle/test_fp8_fwd_kernel.py
"""

import math

import pytest
import paddle

from flashmask.cute.mxfp8 import quantize_qkv, quantize_qkv_rowwise
from flashmask.cute.ops.fwd import _flash_attn_fwd
from tests.paddle.generate_startend_row_indices import (
    generate_none_mask,
    generate_sliding_window_mask,
    generate_causal_document_mask,
)
from tests.paddle.test_fp8_fake_quant import (
    Recipe, fake_attention, metrics, _dense_bias, _to_bhsd, _to_bshd,
)

MX_TAU = math.log2(448.0 / 256.0)   # the kernel's rescale_threshold (E4M3 max / P scale)

MASKS = {
    "full": lambda b, sq, sk, h: generate_none_mask(b, sq, sk, h, causal=False),
    "causal": lambda b, sq, sk, h: generate_none_mask(b, sq, sk, h, causal=True),
    "causal_document": generate_causal_document_mask,
    "sliding_window": generate_sliding_window_mask,
}


@pytest.mark.parametrize("mask", list(MASKS))
@pytest.mark.parametrize("b, s, h, h_kv", [(1, 4096, 4, 2), (2, 1024, 2, 2), (1, 32768, 2, 1)])
@pytest.mark.parametrize("d", [128, 256])
def test_mxfp8_fwd_kernel(d, mask, b, s, h, h_kv):
    paddle.seed(2026)
    g = h // h_kv
    q = paddle.randn([b, s, h, d]).astype("bfloat16")
    k = paddle.randn([b, s, h_kv, d]).astype("bfloat16")
    v = paddle.randn([b, s, h_kv, d]).astype("bfloat16")
    startend, causal = MASKS[mask](b, s, s, 1)
    bias = _dense_bias(startend, causal, s, s)

    mx = quantize_qkv(q, k, v, "float32")
    out, lse = _flash_attn_fwd(
        mx.q, mx.k, mx.v, causal=causal, startend_row_indices=startend, return_lse=True,
        mx_scales=(mx.sf_q, mx.sf_k, mx.sf_v),
    )

    check(f"d={d} {mask} b={b} s={s} h={h}/{h_kv}", out, lse, (q, k, v), (mx.q_dq, mx.k_dq, mx.v_dq),
          g, bias, Recipe(pv=True, tau=MX_TAU))  # re-quantizing the MXFP8 V is idempotent


@pytest.mark.parametrize("mask", list(MASKS))
@pytest.mark.parametrize("b, s, h, h_kv", [(1, 4096, 4, 2), (2, 1024, 2, 2), (1, 32768, 2, 1)])
@pytest.mark.parametrize("d, dv", [(512, 512), (576, 512)])
def test_rowwise_fp8_fwd_kernel(d, dv, mask, b, s, h, h_kv):
    paddle.seed(2026)
    g = h // h_kv
    # Spread the row / key / channel magnitudes so the Q, K and V descales all differ (equal
    # descales would hide an indexing bug; checked by feeding reversed descales: all fail).
    q = (paddle.randn([b, s, h, d]) * paddle.exp(paddle.randn([d]) * 0.5)
         * paddle.exp(paddle.randn([b, s, h, 1]) * 0.5)).astype("bfloat16")
    k = (paddle.randn([b, s, h_kv, d]) * paddle.exp(paddle.randn([b, s, h_kv, 1]))).astype("bfloat16")
    v = (paddle.randn([b, s, h_kv, dv]) * paddle.exp(paddle.randn([dv]))).astype("bfloat16")
    startend, causal = MASKS[mask](b, s, s, 1)
    bias = _dense_bias(startend, causal, s, s)

    fp8 = quantize_qkv_rowwise(q, k, v, "float32")
    out, lse = _flash_attn_fwd(
        fp8.q, fp8.k, fp8.v, causal=causal, startend_row_indices=startend, return_lse=True,
        fp8_descales=(fp8.descale_q, fp8.descale_k, fp8.descale_v),
    )
    check(f"d={d}/{dv} {mask} b={b} s={s} h={h}/{h_kv}", out, lse, (q, k, v),
          (fp8.q_dq, fp8.k_dq, fp8.v_dq), g, bias, Recipe(pv=True, pv_v=False, tau=MX_TAU))


def check(label, out, lse, qkv, qkv_dq, g, bias, recipe):
    """Compare the kernel with the fake-quant model on the very same dequantized operands and
    with FP32 attention on the original BF16 inputs; assert the pass criterion."""
    q, k, v = qkv
    b, s, h, d = q.shape
    zeros = paddle.zeros([b, h, s, v.shape[-1]])
    qd, kd, vd = _to_bhsd(qkv_dq[0]), _to_bhsd(qkv_dq[1], g), _to_bhsd(qkv_dq[2], g)
    o_model = _to_bshd(fake_attention(qd, kd, vd, zeros, bias, recipe)[0])
    o_fp32 = _to_bshd(fake_attention(_to_bhsd(q), _to_bhsd(k, g), _to_bhsd(v, g), zeros, bias,
                                     Recipe(delta="matched"))[0])
    # float64: an FP32 reference is itself off by up to ~17 ULP (1e-3 at |LSE| ~ 1000, s = 32K).
    scores = paddle.matmul(qd.astype("float64"), kd.astype("float64"), transpose_y=True) / d ** 0.5
    if bias is not None:
        scores = scores + bias.astype("float64")
    lse_ref = paddle.logsumexp(scores, axis=-1)                     # [b, h, s]
    del scores

    kernel_vs_model = metrics(out, o_model)
    model_vs_fp32 = metrics(o_model, o_fp32)
    kernel_vs_fp32 = metrics(out, o_fp32)
    lse_err = float((lse.astype("float64") - lse_ref).abs().max())
    print(f"\n[{label}]  cos / relL2 / maxrel")
    for name, m in (("kernel vs model", kernel_vs_model), ("model vs FP32", model_vs_fp32),
                    ("kernel vs FP32", kernel_vs_fp32)):
        print(f"  {name:<16}{m['cos']:>11.7f}{m['relL2']:>10.2e}{m['maxrel']:>10.2e}")
    print(f"  LSE max abs err vs FP32 on dequantized inputs: {lse_err:.2e}")

    assert bool(paddle.isfinite(out.astype("float32")).all())
    assert kernel_vs_model["relL2"] < 0.1 * model_vs_fp32["relL2"]
    assert lse_err < 1e-3
