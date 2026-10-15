# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

try:
    from .flashmask_cuda_utils import accum_zero_axis1_kv, bshd_slice_contiguous_kv
except ImportError:
    accum_zero_axis1_kv = None
    bshd_slice_contiguous_kv = None

__all__ = ["accum_zero_axis1_kv", "bshd_slice_contiguous_kv"]
