# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""Context-parallel comm primitives, torch port.

Mirrors `tests/paddle/distributed/cp_utils.py` one-to-one but built on
``torch.distributed`` instead of paddle fleet: DualChunkSwap for the CP-overlap
test plus the ``balanced_swap`` / rank-ordered contiguous primitives the
cp_balance test needs. The math is identical.
"""

import torch
import torch.distributed as dist


def scatter_balance(input_tensor, group, axis=0, mode="dual_chunk", buckets=None):
    """Split ``input_tensor`` along ``axis`` into this rank's CP share.

    ``dual_chunk``: block ``r`` from the front + block ``2*cp-1-r`` from the back.
    ``balanced_swap``: the chunks listed in ``buckets[rank]`` (the cp_balance
    load-balancing plan), concatenated in that order.
    """
    parallelism = dist.get_world_size(group=group)
    if parallelism == 1:
        return input_tensor.clone()

    rank = dist.get_rank(group=group)
    seq_len = input_tensor.shape[axis]
    if mode == "dual_chunk":
        assert seq_len % (parallelism * 2) == 0, (
            f"Input sequence length {seq_len} can't be divided exactly by "
            f"sequence parallelism * 2 {parallelism * 2}"
        )
        interval = seq_len // parallelism // 2
        chunk_start = input_tensor.narrow(axis, interval * rank, interval)
        chunk_end = input_tensor.narrow(axis, seq_len - interval * (rank + 1), interval)
        result = torch.cat([chunk_start, chunk_end], dim=axis)
    elif mode == "balanced_swap":
        assert buckets is not None and len(buckets) == parallelism
        chunk_size = seq_len // sum(len(bucket) for bucket in buckets)
        result = torch.cat(
            [input_tensor.narrow(axis, idx * chunk_size, chunk_size) for _, idx in buckets[rank]],
            dim=axis,
        )
    else:
        raise ValueError(f"unsupported scatter mode: {mode}")
    # .contiguous() copies out so the (large) source tensor can be freed.
    return result.contiguous()


def all_gather_balance(input_tensor, group, axis=0, mode="dual_chunk", buckets=None):
    """Gather CP shares and restore the original sequence order."""
    parallelism = dist.get_world_size(group=group)
    if parallelism == 1:
        return input_tensor.clone()

    if mode == "dual_chunk":
        chunk_start, chunk_end = torch.chunk(input_tensor, 2, dim=axis)
        chunk_start = chunk_start.contiguous()
        chunk_end = chunk_end.contiguous()
        gathered_start = [torch.empty_like(chunk_start) for _ in range(parallelism)]
        dist.all_gather(gathered_start, chunk_start, group=group)
        gathered_end = [torch.empty_like(chunk_end) for _ in range(parallelism)]
        dist.all_gather(gathered_end, chunk_end, group=group)
        gathered_end = gathered_end[::-1]
        return torch.cat(gathered_start + gathered_end, dim=axis)
    if mode != "balanced_swap":
        raise ValueError(f"unsupported gather mode: {mode}")

    assert buckets is not None and len(buckets) == parallelism
    src = input_tensor.contiguous()
    gathered = [torch.empty_like(src) for _ in range(parallelism)]
    dist.all_gather(gathered, src, group=group)
    chunk_size = input_tensor.shape[axis] // len(buckets[0])
    chunks = [None] * sum(len(bucket) for bucket in buckets)
    for source, bucket in enumerate(buckets):
        for local_pos, (_, original_pos) in enumerate(bucket):
            chunks[int(original_pos)] = gathered[source].narrow(axis, local_pos * chunk_size, chunk_size)
    return torch.cat(chunks, dim=axis)


def all_gather_contiguous(input_tensor, group, axis=0):
    """Gather local tensors in rank order (no reordering)."""
    parallelism = dist.get_world_size(group=group)
    gathered = [torch.empty_like(input_tensor) for _ in range(parallelism)]
    dist.all_gather(gathered, input_tensor.contiguous(), group=group)
    return torch.cat(gathered, dim=axis)


def reduce_scatter_any_axis_balance(input_tensor, axis, group):
    """Balanced reduce-scatter: adjoint of ``all_gather_balance`` (dual-chunk)."""
    parallelism = dist.get_world_size(group=group)
    if parallelism == 1:
        return input_tensor.clone()

    assert input_tensor.shape[axis] % (parallelism * 2) == 0, (
        f"Input sequence length {input_tensor.shape[axis]} can't be "
        f"divided exactly by context parallelism * 2 {parallelism * 2}"
    )

    input_start, input_end = torch.chunk(input_tensor, 2, dim=axis)
    chunks_start = list(torch.chunk(input_start, parallelism, dim=axis))
    chunks_end = list(torch.chunk(input_end, parallelism, dim=axis))
    chunks_end = chunks_end[::-1]
    combined_chunks = [
        torch.cat([s, e], dim=axis).contiguous() for s, e in zip(chunks_start, chunks_end)
    ]
    output_buffers = [torch.empty_like(combined_chunks[0]) for _ in range(parallelism)]
    dist.all_to_all(output_buffers, combined_chunks, group=group)
    return torch.stack(output_buffers, dim=0).sum(dim=0)


def reduce_scatter_any_axis_order(input_tensor, axis, group):
    """Reduce-scatter a rank-ordered full tensor (adjoint of all_gather_contiguous)."""
    parallelism = dist.get_world_size(group=group)
    chunks = list(torch.chunk(input_tensor, parallelism, dim=axis))
    buffers = [torch.empty_like(chunks[0]) for _ in range(parallelism)]
    dist.all_to_all(buffers, [chunk.contiguous() for chunk in chunks], group=group)
    return torch.stack(buffers).float().sum(0).to(input_tensor.dtype)


def preprocess_index_dual_chunks(startend_row_indices, chunk_id_first, chunk_id_second, seq_blocksize, max_seqlen_q):
    """Rebase FlashMask row indices for a rank owning two (front+back) chunks."""
    rows_min_first = chunk_id_first * seq_blocksize
    rows_min_second = chunk_id_second * seq_blocksize

    indices_first = torch.clamp(startend_row_indices - rows_min_first, min=0, max=max_seqlen_q)
    indices_second = torch.clamp(startend_row_indices - rows_min_second, min=0, max=max_seqlen_q)
    indices_second = torch.where(
        indices_second != 0, indices_second + max_seqlen_q, indices_second
    )
    return torch.maximum(indices_first, indices_second)
