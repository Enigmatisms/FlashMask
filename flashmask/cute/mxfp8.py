# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""Quantize BF16 Q/K/V for the FP8 FlashMask forwards (paddle): MXFP8 (head_dim <= 256) and
row-wise FP8 (head_dim > 256).

MXFP8 = E4M3 payload + one E8M0 (power-of-two) scale per 32 consecutive values along the
contraction dim of the GEMM that consumes the tensor (OCP MX spec, as in Meta LP-FA4):
  Q, K -> along head_dim (QK^T contracts d);  V -> along the keys (PV contracts the keys).
Scale = 2^ceil(log2(amax / 448)) ("RCEIL", LP-FA4's default; 1 for an all-zero block, at least
2^-126), payload = RN-satfinite(x / scale).
The kernel reads the scales through TMA in the tcgen05 scale-factor layout
(cutlass.utils.blockscaled_layout.tile_atom_to_shape_SF): for a (rows, K) operand, each
128-row x 4-group atom is 512 contiguous bytes, byte = r0 * 16 + r1 * 4 + kk for row
r = r1 * 32 + r0 and scale group kk; atoms are ordered (batch * heads, row atom, K atom)
with the K atom fastest. The quantizers write each scale at `_sf_offset` of that layout.

Row-wise FP8 = E4M3 payload + one FP32 descale amax / 448 (1 if amax = 0) per Q row, per key
(K row) and per V column (dv channel, over all keys); the attention kernel applies them outside
the tensor core. payload = RN-satfinite(x * RN(1 / descale)) (for MXFP8 that equals x / scale).
All roundings are IEEE round-to-nearest, so a numpy reference reproduces every bit.

Small CuTe DSL kernels quantize each tensor and write the scales in the layout the attention
kernel reads: `_quant_rows` for Q and K (one pass; the scale groups lie along a row, reduced by
warp shuffles) and `_quant_cols` for V (a thread owns 4 channels x 32 keys; one pass for MXFP8,
a column-amax pass first for row-wise).
"""

import math
from typing import NamedTuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import paddle
from cutlass import Float32, Int32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op

from flashmask.backend import get_backend
from flashmask.cute import utils

E4M3_MAX = 448.0
BLOCK = 32
_THREADS = 128  # threads per CTA
_CVEC = 4       # V: channels per thread (8 B BF16 loads; a warp reads 256 B per key)


# ----------------------------------------------------------------------------
# Kernels
# ----------------------------------------------------------------------------
@dsl_user_op
def _pow2_ceil(x: Float32, *, loc=None, ip=None) -> Float32:
    """Smallest power of two >= x for an FP32 x > 0, exact (2^-126 for a subnormal x)."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 b, m;\n\t.reg .pred p;\n\tmov.b32 b, $1;\n\tand.b32 m, b, 0x7fffff;\n\t"
            "and.b32 b, b, 0x7f800000;\n\tsetp.ne.b32 p, m, 0;\n\t@p add.s32 b, b, 0x800000;\n\t"
            "mov.b32 $0, b;\n\t}",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _div_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """IEEE (round-to-nearest) a / b; the DSL's `/` lowers to an approximate division."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "div.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def _descale(amax: Float32, pow2: cutlass.Constexpr) -> Float32:
    """amax / 448, rounded up to a power of two if `pow2`; 1 for amax = 0."""
    s = _div_rn(amax, Float32(E4M3_MAX))
    if const_expr(pow2):
        s = _pow2_ceil(s)
    if amax == Float32(0.0):
        s = Float32(1.0)
    return s


@dsl_user_op
def _rcp_rn(x: Float32, *, loc=None, ip=None) -> Float32:
    """IEEE (round-to-nearest) 1 / x; exact for a power of two."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "rcp.rn.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def _quantize(x, d, inv, mQ, mDq, t):
    """Store E4M3 RN-satfinite(x * inv) to chunk t of the flat mQ and, if given, payload * d to
    chunk t of mDq (chunk = cute.size(x) values; inv = RN(1 / d), so x * inv = x / d exactly for a
    power-of-two d). The E4M3 converts run on vectors padded to a multiple of 4 (other lengths,
    e.g. 2 or 18, do not lower)."""
    n = cute.size(x)
    m = (n + 3) // 4 * 4
    xp = cute.make_fragment(m, Float32)  # x * inv, zero padded
    dp = cute.make_fragment(m, Float32)  # d, one padded
    for i in range(m):
        xp[i] = x[i] * inv[i] if i < n else Float32(0.0)
        dp[i] = d[i] if i < n else Float32(1.0)
    q = cute.make_fragment(m, cutlass.Float8E4M3FN)
    q.store(xp.load().to(cutlass.Float8E4M3FN))
    head = lambda f: cute.make_tensor(f.iterator, cute.make_layout(n))
    cute.autovec_copy(head(q), cute.local_tile(mQ, (n,), (t,)))
    if const_expr(mDq is not None):
        dq = cute.make_fragment(m, mDq.element_type)
        val = q.load().to(Float32) * dp.load()
        dq.store(val if const_expr(mDq.element_type is Float32) else val.to(mDq.element_type))
        cute.autovec_copy(head(dq), cute.local_tile(mDq, (n,), (t,)))


@dsl_user_op
def _e8m0_code(x: Float32, *, loc=None, ip=None) -> Int32:
    """E8M0 code (the exponent field) of an FP32 power of two."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 b;\n\tmov.b32 b, $1;\n\tshr.u32 $0, b, 23;\n\t}",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _red_max_abs(ptr: cute.Pointer, x: Float32, *, loc=None, ip=None) -> None:
    """*ptr = max(*ptr, |x|) on FP32 bits (unsigned compare = float compare for |x|)."""
    llvm.inline_asm(
        None,
        [ptr.toint(loc=loc, ip=ip).ir_value(), Float32(x).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b32 b;\n\tmov.b32 b, $1;\n\tand.b32 b, b, 0x7fffffff;\n\t"
        "red.global.max.u32 [$0], b;\n\t}",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def _sf_offset(L, row, group, rows, groups):
    """Byte offset of scale (L, row, group) in the tcgen05 scale-factor layout (module docstring)."""
    atom = (L * (rows // 128) + row // 128) * (groups // 4) + group // 4
    return atom * 512 + (row % 32) * 16 + (row // 32) % 4 * 4 + group % 4


@cute.kernel
def _quant_rows_kernel(mX, mQ, mDq, mScale, row0: Int32, seqlen: Int32, heads: Int32, cols: cutlass.Constexpr,
                       vec: cutlass.Constexpr, chunks: cutlass.Constexpr, pow2: cutlass.Constexpr):
    """Flat mX (b * seqlen * heads rows of `cols`). pow2 (MXFP8): one descale per 32 values, mScale
    is the packed E8M0 buffer of (batch * head, token, group). Else one FP32 descale per row,
    mScale is [b, heads, seqlen]. A warp moves `chunks` runs of 32 * vec contiguous values
    (coalesced); in each, lane j owns values [j * vec, (j + 1) * vec) and the group // vec lanes
    of a group reduce its amax by shuffles (all lanes take part, so invalid tail lanes load zeros)."""
    group = BLOCK if const_expr(pow2) else cols
    t = cute.arch.block_idx()[0] * _THREADS + cute.arch.thread_idx()[0]
    raw = cute.make_fragment((vec, chunks), mX.element_type)
    raw.fill(0)
    for c in cutlass.range_constexpr(chunks):  # all loads first
        u = (t // 32 * chunks + c) * 32 + t % 32
        if u < cute.size(mX) // vec:
            cute.autovec_copy(cute.local_tile(mX, (vec,), (u,)), raw[None, c])
    for c in cutlass.range_constexpr(chunks):
        u = (t // 32 * chunks + c) * 32 + t % 32
        x = cute.make_fragment(vec, Float32)
        x.store(raw[None, c].load().to(Float32))
        amax = Float32(0.0)
        for i in cutlass.range_constexpr(vec):
            amax = cute.arch.fmax(amax, cute.arch.fmax(x[i], -x[i]))
        s = _descale(utils.warp_reduce(amax, cute.arch.fmax, width=group // vec), pow2)
        if u < cute.size(mX) // vec:
            if u % (group // vec) == 0:
                row = row0 + u * vec // cols  # global row: mX/mQ/mDq start at row row0
                L, token = row // (heads * seqlen) * heads + row % heads, row // heads % seqlen
                if const_expr(pow2):
                    mScale[_sf_offset(L, token, u * vec % cols // BLOCK, seqlen, cols // BLOCK)] = cutlass.Uint8(_e8m0_code(s))
                else:
                    mScale[L * seqlen + token] = s
            d = cute.make_fragment(vec, Float32)
            d.fill(s)
            inv = cute.make_fragment(vec, Float32)
            inv.fill(_rcp_rn(s))
            _quantize(x, d, inv, mQ, mDq, u)


def _load_block(mX, rb, cv, cols):
    """Columns [cv * _CVEC, (cv + 1) * _CVEC) of rows [rb * BLOCK, (rb + 1) * BLOCK) of the flat
    mX (rows of `cols`): a (_CVEC, BLOCK) fragment, one vector load per row, all issued first."""
    nv = cols // _CVEC
    raw = cute.make_fragment((_CVEC, BLOCK), mX.element_type)
    for i in range(BLOCK):
        cute.autovec_copy(cute.local_tile(mX, (_CVEC,), ((rb * BLOCK + i) * nv + cv,)), raw[None, i])
    return raw


def _block_amax(raw, amax):
    """amax[j] = max(amax[j], max_i |raw[j, i]|)."""
    for j in range(_CVEC):
        a = amax[j]
        for i in range(BLOCK):
            x = Float32(raw[j, i])
            a = cute.arch.fmax(a, cute.arch.fmax(x, -x))
        amax[j] = a


@cute.kernel
def _col_amax_kernel(mX, mAmax, rbase: Int32, seqlen: Int32, blocks: Int32, cols: cutlass.Constexpr):
    """Flat mX (rows of `cols`, starting at global key block rbase) -> mAmax[batch, c] = max over
    the batch's keys of |x[r, c]|. Thread t owns _CVEC columns of `blocks` consecutive BLOCK-row
    blocks (blocks divides seqlen / BLOCK) and red.max-es their amax into the zeroed mAmax:
    blocks x fewer atomics on the same b * cols addresses than one per block."""
    t = cute.arch.block_idx()[0] * _THREADS + cute.arch.thread_idx()[0]
    nv = cols // _CVEC
    if t < cute.size(mX) // (_CVEC * BLOCK * blocks):
        rb0, cv = t // nv * blocks, t % nv
        amax = cute.make_fragment(_CVEC, Float32)
        amax.fill(0.0)
        for k in cutlass.range(blocks, unroll=1):
            _block_amax(_load_block(mX, rb0 + k, cv, cols), amax)
        out = ((rbase + rb0) * BLOCK // seqlen * nv + cv) * _CVEC
        for j in cutlass.range_constexpr(_CVEC):
            _red_max_abs(utils.elem_pointer_i64(mAmax, (out + j,)), amax[j])


@cute.kernel
def _quant_cols_kernel(mX, mQ, mDq, mAmax, mScale, rbase: Int32, seqlen: Int32, cols: cutlass.Constexpr,
                       dv: cutlass.Constexpr, pow2: cutlass.Constexpr):
    """Flat mX (b * seqlen, cols = heads * dv); thread t owns a (_CVEC columns x BLOCK keys) block.
    pow2 (MXFP8): the block is a scale group; its amax is reduced here (one pass) and mScale is the
    packed E8M0 buffer of (batch * head, channel, key block). Else (row-wise) each column has one
    FP32 descale from mAmax[batch, c] (_col_amax_kernel), written to mScale by the batch's first
    block. mX / mQ / mDq start at global key block rbase; scale indices are global."""
    t = cute.arch.block_idx()[0] * _THREADS + cute.arch.thread_idx()[0]
    if t < cute.size(mX) // (_CVEC * BLOCK):
        nv = cols // _CVEC
        rb, cv = t // nv, t % nv
        raw = _load_block(mX, rb, cv, cols)
        d = cute.make_fragment(_CVEC, Float32)
        if const_expr(pow2):
            amax = cute.make_fragment(_CVEC, Float32)
            amax.fill(0.0)
            _block_amax(raw, amax)
            rg = (rbase + rb) * BLOCK                       # global key row of the block
            L, k = rg // seqlen * (cols // dv), rg % seqlen // BLOCK
            for j in cutlass.range_constexpr(_CVEC):
                d[j] = _descale(amax[j], True)
                c = cv * _CVEC + j
                mScale[_sf_offset(L + c // dv, c % dv, k, dv, seqlen // BLOCK)] = cutlass.Uint8(_e8m0_code(d[j]))
        else:
            rg = (rbase + rb) * BLOCK
            out = rg // seqlen * nv + cv
            cute.autovec_copy(cute.local_tile(mAmax, (_CVEC,), (out,)), d)
            for j in cutlass.range_constexpr(_CVEC):
                d[j] = _descale(d[j], False)
            if rg % seqlen == 0:
                cute.autovec_copy(d, cute.local_tile(mScale, (_CVEC,), (out,)))
        inv = cute.make_fragment(_CVEC, Float32)
        for j in cutlass.range_constexpr(_CVEC):
            inv[j] = _rcp_rn(d[j])
        for i in cutlass.range_constexpr(BLOCK):
            x = cute.make_fragment(_CVEC, Float32)
            x.store(raw[None, i].load().to(Float32))
            _quantize(x, d, inv, mQ, mDq, (rb * BLOCK + i) * nv + cv)


def _flat(p, size):
    return None if p is None else cute.make_tensor(p, cute.make_layout(size))


@cute.jit
def _quant_rows_launch(pX, pQ, pDq, pScale, n: Int32, n_scale: Int32, row0: Int32, seqlen: Int32,
                       heads: Int32, cols: cutlass.Constexpr, vec: cutlass.Constexpr,
                       chunks: cutlass.Constexpr, pow2: cutlass.Constexpr, stream: cuda.CUstream):
    """n values from global row row0 (pointers already offset); mScale: the whole scale buffer."""
    mX, mQ, mDq, mScale = _flat(pX, n), _flat(pQ, n), _flat(pDq, n), _flat(pScale, n_scale)
    per_cta = _THREADS * vec * chunks
    _quant_rows_kernel(mX, mQ, mDq, mScale, row0, seqlen, heads, cols, vec, chunks, pow2).launch(
        grid=[(n + per_cta - 1) // per_cta, 1, 1], block=[_THREADS, 1, 1], stream=stream)


@cute.jit
def _quant_cols_launch(pX, pQ, pDq, pAmax, pScale, n: Int32, n_scale: Int32, rbase: Int32,
                       seqlen: Int32, blocks: Int32, cols: cutlass.Constexpr, dv: cutlass.Constexpr,
                       pow2: cutlass.Constexpr, amax_pass: cutlass.Constexpr, stream: cuda.CUstream):
    """n values from global key block rbase (pointers already offset); mAmax / mScale: whole
    buffers. amax_pass: the row-wise column-amax pass, else the quantize pass."""
    mX, mQ, mDq = _flat(pX, n), _flat(pQ, n), _flat(pDq, n)
    mAmax, mScale = _flat(pAmax, n_scale), _flat(pScale, n_scale)  # row-wise amax: [b, cols]
    if const_expr(amax_pass):
        threads = n // (_CVEC * BLOCK * blocks)
        _col_amax_kernel(mX, mAmax, rbase, seqlen, blocks, cols).launch(
            grid=[(threads + _THREADS - 1) // _THREADS, 1, 1], block=[_THREADS, 1, 1], stream=stream)
    else:
        threads = n // (_CVEC * BLOCK)
        _quant_cols_kernel(mX, mQ, mDq, mAmax, mScale, rbase, seqlen, cols, dv, pow2).launch(
            grid=[(threads + _THREADS - 1) // _THREADS, 1, 1], block=[_THREADS, 1, 1], stream=stream)


_compiled = {}
_DTYPES = {
    paddle.bfloat16: cutlass.BFloat16, paddle.float16: cutlass.Float16, paddle.float32: Float32,
    paddle.float8_e4m3fn: cutlass.Float8E4M3FN, paddle.uint8: cutlass.Uint8,
}


def _ptr(t, offset=0):
    """Raw gmem pointer of a paddle tensor, `offset` elements in (much cheaper per call than
    from_dlpack)."""
    if t is None:
        return None
    dtype = _DTYPES[t.dtype]
    addr = t.data_ptr() + offset * dtype.width // 8
    assert addr % 16 == 0, "the quantizers move 16 B vectors"
    return make_ptr(dtype, addr, cute.AddressSpace.gmem, assumed_align=16)


# The kernels index in Int32: larger tensors run in chunks of whole rows of at most this many values.
_CHUNK = 1 << 30


def _run(launch, key, stream, args, *runtime, constexprs=()):
    """Compile `launch` once per key, then run it on `stream` (args: pointers, runtime: Int32
    values). Host cost matters (it is paid per layer per step): paddle's Tensor.size / reshape
    cost ~3-4 us each, so callers compute shapes from python ints and allocate outputs in their
    final shape."""
    runtime = [Int32(v) for v in runtime]
    if key not in _compiled:
        _compiled[key] = cute.compile(launch, *args, *runtime, *constexprs, stream)
    _compiled[key](*args, *runtime, stream)


def _stream(t):
    return cuda.CUstream(get_backend(t).cuda_stream())


def _empty(shape, dtype):
    return None if dtype is None else paddle.empty(shape, dtype)


def _quant_rows(x, pow2, dequant_dtype):
    """x [b, s, h, C]: one descale per 32 values of a row if `pow2` (MXFP8), else per row.
    Returns (E4M3 payload, scales, dequantized copy or None); scales are the packed E8M0 bytes of
    (b * h, s, C / 32) if `pow2`, else FP32 [b, h, s]."""
    shape = x.shape
    b, s, h, C = shape
    # Values per lane: MXFP8 16 (2 lanes per group, 2 runs per warp), row-wise C // 32 (a warp per
    # row). Measured (ncu, Q d=128 S=32K h=16) against 8 or 32 per lane: 8 spends ~1.6x the
    # instructions on per-group work, 32 strides lanes 64 B apart and loses DRAM throughput.
    vec, chunks = (16, 2) if pow2 else (C // 32, 1)
    assert C % 32 == 0, f"unsupported row length {C}"
    rows = b * s * h
    x = x.contiguous()
    q, dq = paddle.empty(shape, "float8_e4m3fn"), _empty(shape, dequant_dtype)
    if pow2:
        assert s % 128 == 0 and C % 128 == 0, "MXFP8 scale layout needs s % 128, d % 128"
        n_scale, scale = rows * C // BLOCK, paddle.empty([rows * C // BLOCK], "uint8")
    else:
        n_scale, scale = rows, paddle.empty([b, h, s], "float32")
    step = _CHUNK // C                                # rows per launch (any row boundary works)
    for r0 in range(0, rows, step):
        off, n = r0 * C, min(step, rows - r0) * C
        _run(_quant_rows_launch, ("rows", x.dtype, dequant_dtype, C, pow2), _stream(x),
             (_ptr(x, off), _ptr(q, off), _ptr(dq, off), _ptr(scale)), n, n_scale, r0, s, h,
             constexprs=(C, vec, chunks, pow2))
    return q, scale, dq


def _quant_cols(v, pow2, dequant_dtype):
    """v [b, s, h, dv]: one descale per dv channel and 32-key block if `pow2` (MXFP8), else per dv
    channel over all s keys. Returns (E4M3 payload, scales, dequantized copy or None); scales are
    the packed E8M0 bytes of (b * h, dv, s / 32) if `pow2`, else FP32 [b, h, dv]. MXFP8 is one
    pass; row-wise needs a column-amax pass first."""
    shape = v.shape
    b, s, h, dv = shape
    assert s % BLOCK == 0 and dv % _CVEC == 0, f"unsupported V {shape}"
    blocks = math.gcd(s // BLOCK, 4)  # row-wise amax: 32-key blocks per thread (best of 1/4/8/16)
    cols = h * dv
    v = v.contiguous()
    q, dq = paddle.empty(shape, "float8_e4m3fn"), _empty(shape, dequant_dtype)
    if pow2:
        assert s % 128 == 0 and dv % 128 == 0, "MXFP8 scale layout needs s % 128, dv % 128"
        n_scale = b * s * cols // BLOCK
        amax, scale = None, paddle.empty([n_scale], "uint8")
    else:
        n_scale = b * cols
        amax, scale = paddle.zeros([b, h, dv], "float32"), paddle.empty([b, h, dv], "float32")
    # Chunks start on a multiple of blocks * BLOCK keys (the row-wise amax thread granule); the
    # row-wise amax spans all chunks, so every amax chunk runs before any quantize chunk.
    step = _CHUNK // cols // 128 * 128
    stream = _stream(v)
    for amax_pass in (False,) if pow2 else (True, False):
        for r0 in range(0, b * s, step):
            off, n = r0 * cols, min(step, b * s - r0) * cols
            _run(_quant_cols_launch, ("cols", v.dtype, dequant_dtype, cols, dv, pow2, amax_pass), stream,
                 (_ptr(v, off), _ptr(q, off), _ptr(dq, off), _ptr(amax), _ptr(scale)),
                 n, n_scale, r0 // BLOCK, s, blocks, constexprs=(cols, dv, pow2, amax_pass))
    return q, scale, dq


# ----------------------------------------------------------------------------
# MXFP8 (head_dim <= 256)
# ----------------------------------------------------------------------------
class MxfpInputs(NamedTuple):
    q: paddle.Tensor        # E4M3 [b, s_q, h, d]
    k: paddle.Tensor        # E4M3 [b, s_k, h_k, d]
    v: paddle.Tensor        # E4M3 [b, s_k, h_k, dv]
    sf_q: paddle.Tensor     # packed scales, rows = s_q, K = d,   per (b, h)
    sf_k: paddle.Tensor     # packed scales, rows = s_k, K = d,   per (b, h_k)
    sf_v: paddle.Tensor     # packed scales, rows = dv,  K = s_k, per (b, h_k)
    q_dq: paddle.Tensor     # dequantized copies (exactly what the kernel multiplies), or None
    k_dq: paddle.Tensor
    v_dq: paddle.Tensor


def quantize_qkv(q, k, v, dequant_dtype=None):
    """Quantize BF16 [b, s, h, d] Q/K/V for the MXFP8 forward. Batch-head index = b * h + h_i."""
    q8, sf_q, q_dq = _quant_rows(q, True, dequant_dtype)
    k8, sf_k, k_dq = _quant_rows(k, True, dequant_dtype)
    v8, sf_v, v_dq = _quant_cols(v, True, dequant_dtype)
    return MxfpInputs(q8, k8, v8, sf_q, sf_k, sf_v, q_dq, k_dq, v_dq)


# ----------------------------------------------------------------------------
# Row-wise FP8 (head_dim > 256)
# ----------------------------------------------------------------------------
class Fp8RowwiseInputs(NamedTuple):
    q: paddle.Tensor        # E4M3 [b, s_q, h, d]
    k: paddle.Tensor        # E4M3 [b, s_k, h_k, d]
    v: paddle.Tensor        # E4M3 [b, s_k, h_k, dv]
    descale_q: paddle.Tensor  # FP32 [b, h, s_q]: one per query row
    descale_k: paddle.Tensor  # FP32 [b, h_k, s_k]: one per key
    descale_v: paddle.Tensor  # FP32 [b, h_k, dv]: one per dv channel (over all keys)
    q_dq: paddle.Tensor     # dequantized copies (payload * descale), or None
    k_dq: paddle.Tensor
    v_dq: paddle.Tensor


def quantize_qkv_rowwise(q, k, v, dequant_dtype=None):
    """Quantize BF16 [b, s, h, d] Q/K/V for the row-wise FP8 forward. The descales are
    applied outside the tensor core, so they can be FP32 and exact (see flash_fwd_sm100)."""
    q8, dq, q_dq = _quant_rows(q, False, dequant_dtype)               # dq: [b, h, s_q]
    k8, dk, k_dq = _quant_rows(k, False, dequant_dtype)
    v8, dv, v_dq = _quant_cols(v, False, dequant_dtype)               # dv: [b, h_k, dv]
    return Fp8RowwiseInputs(q8, k8, v8, dq, dk, dv, q_dq, k_dq, v_dq)


def quantize_fp8(q, k, v):
    """Inputs of the FP8 forward for BF16 q/k/v: MXFP8 for head_dim <= 256, row-wise FP8 above.
    Returns the E4M3 (q, k, v), the `_flash_attn_fwd` keyword carrying their scales, and the
    dequantized (q, k, v) in the input dtype, i.e. the values the forward computes with (exact
    for MXFP8; row-wise FP8 rounds payload * FP32 descale once to BF16)."""
    if q.shape[-1] <= 256:
        x = quantize_qkv(q, k, v, q.dtype)
        scales = dict(mx_scales=(x.sf_q, x.sf_k, x.sf_v))
    else:
        x = quantize_qkv_rowwise(q, k, v, q.dtype)
        scales = dict(fp8_descales=(x.descale_q, x.descale_k, x.descale_v))
    return (x.q, x.k, x.v), scales, (x.q_dq, x.k_dq, x.v_dq)
