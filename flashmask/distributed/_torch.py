# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""Torch implementation of the collective contract."""

from contextlib import contextmanager

import numpy as np
import torch
import torch.distributed as dist

from .base import DistBase


class TorchDist(DistBase):
    name = "torch"

    def world_size(self, group=None):
        return int(dist.get_world_size(group=group))

    def rank(self, group=None):
        return int(dist.get_rank(group=group))

    def global_ranks(self, group=None):
        if group is None:
            return list(range(self.world_size()))
        return list(dist.get_process_group_ranks(group))

    def broadcast_bytes(self, buf, src, group=None, be=None):
        # NCCL cannot broadcast host memory, so the id has to land on the device
        # the caller is using. ``be`` carries that device; without it, fall back
        # to torch's current device.
        device = be.device if be is not None else torch.cuda.current_device()
        arr = np.frombuffer(bytes(buf), dtype=np.uint8).copy()
        t = torch.from_numpy(arr).to(device=device)
        dist.broadcast(t, src=src, group=group)
        return bytes(t.cpu().numpy().tobytes())

    def all_to_all(self, out_tensor_list, in_tensor_list, group=None):
        # torch's list form matches paddle's; all_to_all_single would need the
        # caller to flatten and pass split sizes, which the callers do not do.
        dist.all_to_all(out_tensor_list, in_tensor_list, group=group)
        return out_tensor_list

    @contextmanager
    def nvtx_range(self, label):
        torch.cuda.nvtx.range_push(label)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()

    def nvtx_range_push(self, label):
        torch.cuda.nvtx.range_push(label)

    def nvtx_range_pop(self):
        torch.cuda.nvtx.range_pop()
