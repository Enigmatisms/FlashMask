# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""Paddle implementation of the collective contract."""

from contextlib import contextmanager

import numpy as np
import paddle
import paddle.distributed as dist

from .base import DistBase


class PaddleDist(DistBase):
    name = "paddle"

    def world_size(self, group=None):
        return int(group.world_size) if group is not None else int(dist.get_world_size())

    def rank(self, group=None):
        return int(group.rank) if group is not None else int(dist.get_rank())

    def global_ranks(self, group=None):
        if group is not None:
            return list(group.ranks)
        return list(range(self.world_size()))

    def broadcast_bytes(self, buf, src, group=None, be=None):
        arr = np.frombuffer(bytes(buf), dtype=np.uint8).copy()
        t = paddle.to_tensor(arr, dtype="uint8")
        dist.broadcast(t, src=src, group=group)
        return bytes(t.numpy().tobytes())

    def all_to_all(self, out_tensor_list, in_tensor_list, group=None):
        dist.alltoall(
            out_tensor_list=out_tensor_list, in_tensor_list=in_tensor_list, group=group
        )
        return out_tensor_list

    @contextmanager
    def nvtx_range(self, label):
        paddle.base.core.nvprof_nvtx_push(label)
        try:
            yield
        finally:
            paddle.base.core.nvprof_nvtx_pop()

    def nvtx_range_push(self, label):
        paddle.base.core.nvprof_nvtx_push(label)

    def nvtx_range_pop(self):
        paddle.base.core.nvprof_nvtx_pop()
