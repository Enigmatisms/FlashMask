# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""Timing + FLOP helpers for the FlashMask benchmarks.

Framework-neutral on purpose: the caller passes its own ``sync`` (device
synchronize) and, if it benchmarks a backward pass, a ``before_each`` hook to
clear gradients. Timing is a host timer around a synchronised region, which is
accurate enough at the millisecond scale these kernels run at and avoids
depending on either framework's event API.
"""

import gc
import time

import numpy as np

__all__ = ["do_bench", "cal_flops", "cal_tflops"]


def cal_flops(B, H, Sq, Sk, D, DV, mode="fwd"):
    """Dense attention FLOPs. A mask only ever removes work, so TFLOPS computed
    from this is the *effective* dense-equivalent throughput."""
    assert mode in ("fwd", "bwd", "fwd_bwd")
    if mode == "fwd":
        return 2 * B * Sq * Sk * H * (D + DV)
    if mode == "bwd":
        return 2 * B * Sq * Sk * H * (3 * D + 2 * DV)
    return 2 * B * Sq * Sk * H * (4 * D + 3 * DV)


def cal_tflops(flops, time_ms):
    return flops * (1e3 / time_ms) / 1e12


def do_bench(fn, sync, warmup=10, rep=50, before_each=None, return_mode="mean"):
    """Time ``fn`` in milliseconds.

    Args:
        fn: callable to benchmark, no arguments.
        sync: device synchronize, e.g. ``paddle.device.synchronize`` or
            ``torch.cuda.synchronize``.
        warmup: untimed iterations first (kernel JIT / autotune settle here).
        rep: timed iterations.
        before_each: optional hook run before each timed iteration and NOT
            counted, e.g. clearing gradients so a backward does not accumulate.
        return_mode: "min", "max", "mean", "median" or "all".

    Returns:
        float milliseconds (or a list when ``return_mode="all"``).
    """
    assert return_mode in ("min", "max", "mean", "median", "all")

    # One eager call first so compilation is never inside a timed region.
    fn()
    sync()
    for _ in range(warmup):
        fn()
    sync()

    times = []
    gc.collect()
    gc.disable()
    try:
        for _ in range(rep):
            if before_each is not None:
                before_each()
            sync()
            start = time.perf_counter()
            fn()
            sync()
            times.append((time.perf_counter() - start) * 1e3)
    finally:
        gc.enable()

    if return_mode == "all":
        return times
    return float(getattr(np, return_mode)(times))
