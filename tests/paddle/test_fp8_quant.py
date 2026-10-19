# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""The CuTe DSL quantizers in flashmask.cute.mxfp8 vs an independent reference of the same formats
(IEEE FP32 arithmetic in numpy, E4M3 rounding by paddle's cast): payloads, scales and dequantized
copies must be bit-identical. (Paddle's own FP32 divide is not correctly rounded, so the reference
does its arithmetic in numpy.)

Run:  pytest -s tests/paddle/test_fp8_quant.py
"""

import numpy as np
import pytest
import paddle

from flashmask.cute.mxfp8 import E4M3_MAX, BLOCK, quantize_qkv, quantize_qkv_rowwise

F32 = np.float32


def pack_scales(sf):
    """sf: uint8 [L, rows, K / 32] -> flat uint8 in the tcgen05 scale-factor TMA layout."""
    L, rows, groups = sf.shape
    t = sf.reshape([L, rows // 128, 4, 32, groups // 4, 4])   # row = (atom, r1, r0), group = (atom, kk)
    return t.transpose([0, 1, 4, 3, 2, 5]).flatten()           # (L, row atom, K atom, r0, r1, kk)


def _payload_dq(quotient, descale, dequant_dtype):
    """E4M3 RN(quotient) as a paddle tensor, and payload * descale (FP32) in dequant_dtype."""
    payload = paddle.to_tensor(quotient).astype("float8_e4m3fn")
    if dequant_dtype is None:
        return payload, None
    dq = (payload.astype("float32").numpy() * descale).astype(F32)
    return payload, paddle.to_tensor(dq).astype(dequant_dtype)


def _ref_mx(x, axis, dequant_dtype):
    """MXFP8 along `axis`: (payload, E8M0 bytes with `axis` shrunk by 32, dequantized or None)."""
    xt = np.moveaxis(x.astype("float32").numpy(), axis, -1)
    blk = xt.reshape(*xt.shape[:-1], xt.shape[-1] // BLOCK, BLOCK)
    amax = np.abs(blk).max(-1, keepdims=True)
    m, e = np.frexp(amax / F32(E4M3_MAX))                      # ratio = m * 2^e, m in [0.5, 1)
    e = np.clip(np.where(m == 0.5, e - 1, e), -126, 127)       # ceil(log2(ratio)), >= -126
    e = np.where(amax == 0, 0, e)
    scale = np.ldexp(F32(1), e).astype(F32)
    to_x = lambda t: np.moveaxis(t.reshape(xt.shape), -1, axis)
    payload, dq = _payload_dq(to_x(blk / scale), to_x(np.broadcast_to(scale, blk.shape)), dequant_dtype)
    return payload, paddle.to_tensor(np.moveaxis(e[..., 0] + 127, -1, axis).astype(np.uint8)), dq


def _ref_rowwise(x, axis, dequant_dtype):
    """One FP32 descale amax / 448 per slice reduced over `axis`, payload = RN(x * RN(1 / descale)):
    (payload, descale, dequantized)."""
    x = x.astype("float32").numpy()
    amax = np.abs(x).max(axis=axis, keepdims=True)
    descale = np.where(amax == 0, F32(1), amax / F32(E4M3_MAX)).astype(F32)
    quotient = np.clip(x * (F32(1) / descale), -E4M3_MAX, E4M3_MAX).astype(F32)
    payload, dq = _payload_dq(quotient, np.broadcast_to(descale, x.shape), dequant_dtype)
    return payload, paddle.to_tensor(descale.squeeze(axis)), dq


def _inputs(b, s, h, h_kv, d, dv):
    """BF16 Q/K/V with magnitudes spread over ~2^±12 per row and per channel, all-zero rows and
    32-blocks, and a few extreme values (huge, tiny, exactly 448 * 2^k)."""
    paddle.seed(2026)
    spread = lambda *shape: paddle.pow(paddle.full(shape, 2.0), paddle.randn(shape) * 4)
    q = paddle.randn([b, s, h, d]) * spread(b, s, h, 1) * spread(d)
    k = paddle.randn([b, s, h_kv, d]) * spread(b, s, h_kv, 1) * spread(d)
    v = paddle.randn([b, s, h_kv, dv]) * spread(b, s, h_kv, 1) * spread(dv)
    for t in (q, k, v):
        t[0, 3] = 0.0                       # all-zero rows
        t[0, 64:96, 0, :32] = 0.0           # all-zero 32-blocks along both directions
        t[0, 5, 0, 7] = 3.0e37
        t[0, 6, 0, 9] = 1.0e-37
        t[0, 7, 0, 11] = -E4M3_MAX * 2.0 ** -20
    return [t.astype("bfloat16") for t in (q, k, v)]


def _same(a, b):
    assert (a is None) == (b is None)
    if a is not None:
        assert a.dtype == b.dtype and a.shape == b.shape
        assert bool((a.astype("float32") == b.astype("float32")).all())


SHAPES = [(2, 256, 4, 2), (1, 1024, 2, 1)]


@pytest.mark.parametrize("dequant_dtype", [None, "bfloat16", "float32"])
@pytest.mark.parametrize("b, s, h, h_kv", SHAPES)
@pytest.mark.parametrize("d", [128, 256])
def test_mxfp8_quantizer(d, b, s, h, h_kv, dequant_dtype):
    q, k, v = _inputs(b, s, h, h_kv, d, d)
    got = quantize_qkv(q, k, v, dequant_dtype)
    (q8, sq, q_dq), (k8, sk, k_dq) = (_ref_mx(t, -1, dequant_dtype) for t in (q, k))
    v8, sv, v_dq = _ref_mx(v, 1, dequant_dtype)
    bh = lambda t: t.transpose([0, 2, 1, 3]).reshape([-1, t.shape[1], t.shape[3]])
    sf_v = sv.transpose([0, 2, 3, 1]).reshape([b * h_kv, d, s // BLOCK])
    ref = (q8, k8, v8, pack_scales(bh(sq)), pack_scales(bh(sk)), pack_scales(sf_v), q_dq, k_dq, v_dq)
    for a, r in zip(got, ref):
        _same(a, r)


@pytest.mark.parametrize("dequant_dtype", [None, "bfloat16", "float32"])
@pytest.mark.parametrize("b, s, h, h_kv", SHAPES)
@pytest.mark.parametrize("d, dv", [(512, 512), (576, 512)])
def test_rowwise_quantizer(d, dv, b, s, h, h_kv, dequant_dtype):
    q, k, v = _inputs(b, s, h, h_kv, d, dv)
    got = quantize_qkv_rowwise(q, k, v, dequant_dtype)
    (q8, dq, q_dq), (k8, dk, k_dq) = (_ref_rowwise(t, -1, dequant_dtype) for t in (q, k))
    v8, dv_, v_dq = _ref_rowwise(v, 1, dequant_dtype)
    t = lambda x: x.transpose([0, 2, 1])
    for a, r in zip(got, (q8, k8, v8, t(dq), t(dk), dv_, q_dq, k_dq, v_dq)):
        _same(a, r)


@pytest.mark.parametrize("d, dv", [(128, 128), (256, 256), (512, 512), (576, 512)])
def test_chunked_launches(d, dv, monkeypatch):
    """Tensors past the kernels' Int32 range run in chunks of rows; force ~8 chunks per tensor
    on a small input and require the very same bits (payloads, scales, dequantized copies)."""
    from flashmask.cute import mxfp8
    b, s, h, h_kv = 2, 512, 4, 2
    q, k, v = _inputs(b, s, h, h_kv, d, dv)
    quant = quantize_qkv if d <= 256 else quantize_qkv_rowwise
    whole = quant(q, k, v, "bfloat16")
    monkeypatch.setattr(mxfp8, "_CHUNK", 128 * h_kv * dv)  # V: 128 keys per launch
    for a, r in zip(quant(q, k, v, "bfloat16"), whole):
        _same(a, r)
