# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""Host side of the MXFP8 FlashMask forward (paddle).

MXFP8 = E4M3 payload + one E8M0 (power-of-two) scale per 32 consecutive values along the
contraction dim of the GEMM that consumes the tensor (OCP MX spec, as in Meta LP-FA4):
  Q, K -> along head_dim (QK^T contracts d);  V -> along the keys (PV contracts the keys).
Scale = 2^ceil(log2(amax / 448)) ("RCEIL", LP-FA4's default), payload = RN-satfinite(x / scale).

The kernel reads the scales through TMA in the tcgen05 scale-factor layout
(cutlass.utils.blockscaled_layout.tile_atom_to_shape_SF): for a (rows, K) operand, each
128-row x 4-group atom is 512 contiguous bytes, byte = r0 * 16 + r1 * 4 + kk for row
r = r1 * 32 + r0 and scale group kk; atoms are ordered (batch * heads, row atom, K atom)
with the K atom fastest. `pack_scales` produces exactly that.
"""

from typing import NamedTuple

import paddle

E4M3_MAX = 448.0
BLOCK = 32


def _rceil_exponent(amax):
    """Integer e with 2^e = 2^ceil(log2(amax / 448)), exact at powers of two, clamped to E8M0."""
    m, e = paddle.frexp(amax / E4M3_MAX)          # amax / 448 = m * 2^e, m in [0.5, 1)
    e = paddle.where(m == 0.5, e - 1, e)
    return paddle.clip(e, min=-127.0, max=127.0)


def mx_quantize(x, axis):
    """MXFP8-quantize x along `axis`. Returns (E4M3 payload with x's shape,
    E8M0 scale bytes uint8 with `axis` shrunk by 32, dequantized FP32 copy)."""
    xt = paddle.moveaxis(x.astype("float32"), axis, -1)
    shape = xt.shape
    assert shape[-1] % BLOCK == 0, f"dim {axis} ({shape[-1]}) must be a multiple of {BLOCK}"
    blk = xt.reshape(shape[:-1] + [shape[-1] // BLOCK, BLOCK])
    e = _rceil_exponent(blk.abs().max(-1, keepdim=True))
    scale = paddle.pow(paddle.full_like(e, 2.0), e)
    payload = (blk / scale).astype("float8_e4m3fn")
    dequant = payload.astype("float32") * scale
    to_x = lambda t: paddle.moveaxis(t.reshape(shape), -1, axis)
    scale_bytes = paddle.moveaxis((e.squeeze(-1) + 127).astype("uint8"), -1, axis)
    return to_x(payload), scale_bytes, to_x(dequant)


def pack_scales(sf):
    """sf: uint8 [L, rows, K / 32] -> flat uint8 in the tcgen05 scale-factor TMA layout."""
    L, rows, groups = sf.shape
    assert rows % 128 == 0 and groups % 4 == 0, f"scale tile needs rows%128, groups%4: {sf.shape}"
    t = sf.reshape([L, rows // 128, 4, 32, groups // 4, 4])   # row = (atom, r1, r0), group = (atom, kk)
    return t.transpose([0, 1, 4, 3, 2, 5]).flatten()           # (L, row atom, K atom, r0, r1, kk)


class MxfpInputs(NamedTuple):
    q: paddle.Tensor        # E4M3 [b, s_q, h, d]
    k: paddle.Tensor        # E4M3 [b, s_k, h_k, d]
    v: paddle.Tensor        # E4M3 [b, s_k, h_k, dv]
    sf_q: paddle.Tensor     # packed scales, rows = s_q, K = d,   per (b, h)
    sf_k: paddle.Tensor     # packed scales, rows = s_k, K = d,   per (b, h_k)
    sf_v: paddle.Tensor     # packed scales, rows = dv,  K = s_k, per (b, h_k)
    q_dq: paddle.Tensor     # dequantized FP32 copies (exactly what the kernel multiplies)
    k_dq: paddle.Tensor
    v_dq: paddle.Tensor


def quantize_qkv(q, k, v):
    """Quantize BF16 [b, s, h, d] Q/K/V for the MXFP8 forward. Batch-head index = b * h + h_i."""
    b, s_k, h_k, dv = v.shape
    q8, sq, q_dq = mx_quantize(q, axis=-1)                     # sq: [b, s_q, h, d/32]
    k8, sk, k_dq = mx_quantize(k, axis=-1)
    v8, sv, v_dq = mx_quantize(v, axis=1)                      # sv: [b, s_k/32, h_k, dv]
    bh = lambda t: t.transpose([0, 2, 1, 3]).reshape([-1, t.shape[1], t.shape[3]])
    sf_v = sv.transpose([0, 2, 3, 1]).reshape([b * h_k, dv, s_k // BLOCK])
    return MxfpInputs(q8, k8, v8, pack_scales(bh(sq)), pack_scales(bh(sk)), pack_scales(sf_v),
                      q_dq, k_dq, v_dq)
