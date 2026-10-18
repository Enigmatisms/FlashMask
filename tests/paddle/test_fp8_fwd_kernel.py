# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""MXFP8 FlashMask forward kernel vs its fake-quant model (paddle, SM100).

The kernel (flash_fwd_sm100.py, blockscaled=True) must compute exactly what the fake-quant
reference in test_fp8_fake_quant.py describes, up to FP32 accumulation order and the
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

from flashmask.cute.mxfp8 import quantize_qkv
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

    mx = quantize_qkv(q, k, v)
    out, lse = _flash_attn_fwd(
        mx.q, mx.k, mx.v, causal=causal, startend_row_indices=startend, return_lse=True,
        mx_scales=(mx.sf_q, mx.sf_k, mx.sf_v),
    )

    # Fake-quant model on the very same dequantized operands (re-quantizing V is idempotent).
    zeros = paddle.zeros([b, h, s, d])
    qd, kd, vd = _to_bhsd(mx.q_dq), _to_bhsd(mx.k_dq, g), _to_bhsd(mx.v_dq, g)
    o_model = _to_bshd(fake_attention(qd, kd, vd, zeros, bias, Recipe(pv=True, tau=MX_TAU))[0])
    o_fp32 = _to_bshd(fake_attention(_to_bhsd(q), _to_bhsd(k, g), _to_bhsd(v, g), zeros, bias,
                                     Recipe(delta="matched"))[0])
    scores = paddle.matmul(qd, kd, transpose_y=True) / d ** 0.5
    if bias is not None:
        scores = scores + bias
    lse_ref = paddle.logsumexp(scores, axis=-1)                     # [b, h, s]

    kernel_vs_model = metrics(out, o_model)
    model_vs_fp32 = metrics(o_model, o_fp32)
    kernel_vs_fp32 = metrics(out, o_fp32)
    lse_err = float((lse - lse_ref).abs().max())
    print(f"\n[d={d} {mask} b={b} s={s} h={h}/{h_kv}]  cos / relL2 / maxrel")
    for name, m in (("kernel vs model", kernel_vs_model), ("model vs FP32", model_vs_fp32),
                    ("kernel vs FP32", kernel_vs_fp32)):
        print(f"  {name:<16}{m['cos']:>11.7f}{m['relL2']:>10.2e}{m['maxrel']:>10.2e}")
    print(f"  LSE max abs err vs FP32 on dequantized inputs: {lse_err:.2e}")

    assert bool(paddle.isfinite(out.astype("float32")).all())
    assert kernel_vs_model["relL2"] < 0.1 * model_vs_fp32["relL2"]
    assert lse_err < 1e-3
