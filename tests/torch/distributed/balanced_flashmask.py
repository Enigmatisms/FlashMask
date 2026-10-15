# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""Non-overlap FA-v4 FlashMask path for cp_balance testing, torch port.

Mirrors `tests/paddle/distributed/balanced_flashmask.py`: each rank holds its
cp_balance chunk share of q/k/v plus the rebased local mask, gathers the full K/V
in rank order, runs the plain FA-v4 SM100 kernel, and reduce-scatters dK/dV back
to the local share.
"""

import torch
import torch.distributed as dist

from cp_utils import all_gather_contiguous, reduce_scatter_any_axis_order

from flashmask.cute.interface import _flash_attn_bwd, _flash_attn_fwd


def _deterministic():
    return torch.are_deterministic_algorithms_enabled()


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
        local_mask,
        causal=False,
        deterministic=_deterministic(),
    )
    return (
        query_grad,
        reduce_scatter_any_axis_order(key_grad, axis=1, group=group),
        reduce_scatter_any_axis_order(value_grad, axis=1, group=group),
    )


class BalancedFlashMaskFM4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, local_mask, group):
        assert group is not None, "cp_balance path requires an explicit process group"
        output, log_sum_exp = _forward(query, key, value, local_mask, group)
        ctx.save_for_backward(query, key, value, local_mask, output, log_sum_exp)
        ctx.group = group
        return output

    @staticmethod
    def backward(ctx, output_grad):
        query, key, value, local_mask, output, log_sum_exp = ctx.saved_tensors
        query_grad, key_grad, value_grad = _backward(
            query, key, value, local_mask, output, log_sum_exp, output_grad, ctx.group
        )
        return query_grad, key_grad, value_grad, None, None


def balanced_flashmask_attention_fm4(query, key, value, local_mask, group=None):
    assert group is not None, "cp_balance path requires an explicit process group"
    return BalancedFlashMaskFM4.apply(query, key, value, local_mask, group)
