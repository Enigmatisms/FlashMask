# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""Paddle-facing public API.

Signatures are unchanged from when these lived in `cute/interface.py`, including
the keyword arguments that exist only to be asserted at their defaults
(``dropout``, ``rng_name``, ``training`` and friends). They are kept for the
existing callers that pass them; the torch frontend starts from a clean
signature instead.

Kernel orchestration is shared with the torch frontend via `flashmask.cute.ops`.
"""

import paddle

from flashmask.backend import get_backend
from flashmask.cute.ops._common import _check_cutedsl_supported
from .autograd import FlashMaskFunc

__all__ = [
    "flashmask_attention",
]


# TODO: align the parameters with paddle.nn.functional.flashmask_attention?
def flashmask_attention(
    query: paddle.Tensor,
    key: paddle.Tensor,
    value: paddle.Tensor,
    startend_row_indices: paddle.Tensor | None = None,
    *,
    dropout: float = 0.0,
    causal: bool = False,
    window_size: int | tuple | None = None,
    return_softmax_lse: bool = False,
    return_seed_offset: bool = False,
    fixed_seed_offset: paddle.Tensor | None = None,
    rng_name: str = "",
    training: bool = True,
    name: str | None = None,
    softmax_scale: float | None = None,
    block_mask: paddle.Tensor | None = None,
    learnable_sink: paddle.Tensor | None = None,
    group=None,
    fp8: bool = False,
):
    """FlashMask attention on the CuTe DSL kernels.

    fp8: FP8 forward on SM100 (BF16 query/key/value; head_dim = head_dim_v in {128, 256} runs
    MXFP8, head_dim > 256 with 256 < head_dim_v <= 512 runs row-wise FP8; seqlen % 128 == 0, no
    learnable_sink / group). Q/K/V are quantized in the forward and the BF16 backward runs on
    their dequantized values. Off (default): the BF16 kernels, unchanged.
    """
    _check_cutedsl_supported(query, key, value, get_backend(query, key, value))
    assert dropout == 0.0, (
        "flashmask v4 does not support dropout"
    )
    assert not fp8 or query.dtype == key.dtype == value.dtype == paddle.bfloat16, (
        "flashmask v4 fp8=True needs bfloat16 query/key/value"
    )
    # TODO: support sliding-window mask generation when a window_size is given
    assert window_size is None, (
        "flashmask v4 does not support generate sliding window mask automatically"
    )
    assert not return_seed_offset, (
        "flashmask v4 does not support return seed_offset"
    )
    assert fixed_seed_offset is None, (
        "flashmask v4 does not support setting seed_offset"
    )
    assert rng_name == "", (
        "flashmask v4 does not support setting rng_name"
    )
    assert training, (
        "flashmask v4 does not support setting training to False"
    )
    assert name is None, (
        "flashmask v4 does not support setting training name"
    )
    assert block_mask is None, (
        "flashmask v4 does not support block mask"
    )

    if startend_row_indices is not None:
        assert startend_row_indices.dtype == paddle.int32, (
            f"startend_row_indices.dtype must be paddle.int32, but got {startend_row_indices.dtype}"
        )
        assert len(startend_row_indices.shape) == 4, (
            f"startend_row_indices rank must be 4,but got {startend_row_indices.shape}"
        )
        assert startend_row_indices.shape[0] == key.shape[0], (
            f"startend_row_indices.shape[0] must be equal to batch_size, but got {startend_row_indices.shape[0]} and {key.shape[0]}"
        )
        assert startend_row_indices.shape[2] == key.shape[1], (
            f"startend_row_indices.shape[2] must be equal to seqlen_k, but got {startend_row_indices.shape[2]} and {key.shape[1]}"
        )
        assert startend_row_indices.shape[1] in [
            1,
            key.shape[2],
        ], (
            "startend_row_indices head_num must be equal to 1(broadcast) or head_num_k."
        )

        # has_end is not required; it is kept only to make the code easier to reason about
        if causal:
            if startend_row_indices.shape[-1] == 1:
                has_end = False
            elif startend_row_indices.shape[-1] == 2:
                has_end = True
            else:
                raise ValueError(
                    f"Invalid shape of startend_row_indices, when causal is True, the last dimension should be either 1 or 2 but got {startend_row_indices.shape[-1]}"
                )
        else:
            if startend_row_indices.shape[-1] == 2:
                has_end = False
            elif startend_row_indices.shape[-1] == 4:
                has_end = True
            else:
                raise ValueError(
                    f"Invalid shape of startend_row_indices, when causal is False, the last dimension should be either 2 or 4 but got {startend_row_indices.shape[-1]}"
                )

    # when softmax_scale is None, it is set to 1.0 / math.sqrt(head_dim) inside _flash_attn_fwd
    out, lse = FlashMaskFunc.apply(
        query,
        key,
        value,
        causal=causal,
        softmax_scale=softmax_scale,
        learnable_sink=learnable_sink,
        startend_row_indices=startend_row_indices,
        group=group,
        fp8=fp8,
    )
    if return_softmax_lse:
        return [out, lse]
    else:
        return out