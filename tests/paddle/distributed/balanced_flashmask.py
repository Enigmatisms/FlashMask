# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""Non-overlap FA-v4 FlashMask path for cp_balance testing.

The load-balanced (``balance_flashmask_input``) counterpart of the all-gather
reference: each rank holds its cp_balance chunk share of q/k/v plus the rebased
local mask, gathers the full K/V in rank order, runs the plain FA-v4 SM100 kernel,
and reduce-scatters dK/dV back to the local share.
"""

import paddle
from paddle.autograd.py_layer import PyLayer
from paddle.distributed import fleet

from cp_utils import all_gather_contiguous, reduce_scatter_any_axis_order

from flashmask.cute.interface import _flash_attn_bwd, _flash_attn_fwd


def _deterministic():
    return paddle.get_flags(["FLAGS_cudnn_deterministic"])["FLAGS_cudnn_deterministic"]


def _forward(query, key, value, local_mask, group):
    key = all_gather_contiguous(key, group=group, axis=1)
    value = all_gather_contiguous(value, group=group, axis=1)
    return _flash_attn_fwd(
        query,
        key,
        value,
        startend_row_indices=local_mask,
        causal=False,
        return_lse=True,
        pack_gqa=False,
    )


def _backward(query, key, value, local_mask, output, log_sum_exp, output_grad, group):
    key = all_gather_contiguous(key, group=group, axis=1)
    value = all_gather_contiguous(value, group=group, axis=1)
    query_grad, key_grad, value_grad, _ = _flash_attn_bwd(
        query,
        key,
        value,
        output,
        output_grad,
        log_sum_exp,
        flashmask_info=local_mask,
        causal=False,
        deterministic=_deterministic(),
    )
    return (
        query_grad,
        reduce_scatter_any_axis_order(key_grad, axis=1, group=group),
        reduce_scatter_any_axis_order(value_grad, axis=1, group=group),
    )


class BalancedFlashMaskFM4(PyLayer):
    @staticmethod
    def forward(ctx, query, key, value, local_mask, group):
        output, log_sum_exp = _forward(query, key, value, local_mask, group)
        ctx.save_for_backward(query, key, value, local_mask, output, log_sum_exp)
        ctx.group = group
        return output

    @staticmethod
    def backward(ctx, output_grad):
        query, key, value, local_mask, output, log_sum_exp = ctx.saved_tensor()
        return _backward(
            query, key, value, local_mask, output, log_sum_exp, output_grad, ctx.group
        )


def balanced_flashmask_attention_fm4(query, key, value, local_mask, group=None):
    if group is None:
        group = fleet.get_hybrid_communicate_group().get_context_parallel_group()
    return BalancedFlashMaskFM4.apply(query, key, value, local_mask, group)
