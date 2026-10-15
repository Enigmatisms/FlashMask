# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""Collective adapter contract.

Separate from `flashmask.backend` on purpose. The backend is imported by every
attention call; if it pulled in `torch.distributed` / `paddle.distributed`, a
plain single-GPU call would fail on a build without NCCL. Only the code that
actually communicates -- the FM-4 overlap bootstrap and cp_balance -- imports
this.

Kept to what those two callers really need:

- ``world_size`` / ``rank`` / ``global_ranks``: a paddle ``Group`` exposes these
  as attributes (``group.world_size``), a torch ``ProcessGroup`` as functions
  (``dist.get_world_size(group)``), so they cannot be duck-typed.
- ``broadcast_bytes``: distributes the ncclUniqueId produced by
  ``fm4_overlap_get_unique_id``. Overlap builds its own communicator from that
  id and never touches the framework's, so no communicator interop is needed.
- ``all_to_all``: paddle takes lists of tensors, torch's ``all_to_all_single``
  takes one flat tensor plus split sizes. The signature here is the list form;
  the torch implementation adapts.
- ``nvtx_range``: profiler ranges, framework-specific and harmless to skip.
"""

from contextlib import contextmanager

#: Names every concrete dist adapter must provide.
REQUIRED_ATTRS = (
    "name",
    "world_size",
    "rank",
    "global_ranks",
    "broadcast_bytes",
    "all_to_all",
    "nvtx_range",
    "nvtx_range_push",
    "nvtx_range_pop",
)


class DistBase:
    name = None

    @contextmanager
    def nvtx_range(self, label):
        """Scoped profiler range, built on the push/pop pair."""
        self.nvtx_range_push(label)
        try:
            yield
        finally:
            self.nvtx_range_pop()

    def nvtx_range_push(self, label):
        """No-op by default; subclasses push a real profiler range."""

    def nvtx_range_pop(self):
        """No-op by default; subclasses pop a real profiler range."""


def check_dist_class(cls):
    missing = [a for a in REQUIRED_ATTRS if not hasattr(cls, a)]
    if missing:
        raise TypeError(
            f"{cls.__name__} does not implement the distributed contract; "
            f"missing: {', '.join(missing)}"
        )
    return cls
