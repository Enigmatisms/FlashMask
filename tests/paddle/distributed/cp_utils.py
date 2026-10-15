# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""Context-parallel comm primitives, paddle.

DualChunkSwap for the CP-overlap test (each rank owns block ``r`` from the front
of the sequence and block ``2*cp-1-r`` from the back, keeping per-rank work
balanced), plus the ``balanced_swap`` / rank-ordered contiguous primitives the
cp_balance test needs. `tests/torch/distributed/cp_utils.py` is the same math on
``torch.distributed``.
"""

import paddle
from paddle import distributed as dist
from paddle.distributed import fleet


def scatter_balance(input_tensor, group=None, axis=0, mode="dual_chunk", buckets=None):
    """Split ``input_tensor`` along ``axis`` into this rank's CP share.

    ``dual_chunk``: block ``r`` from the front + block ``2*cp-1-r`` from the back.
    ``balanced_swap``: the chunks listed in ``buckets[rank]`` (the cp_balance
    load-balancing plan), concatenated in that order.
    """
    if group is None:
        group = fleet.get_hybrid_communicate_group().get_context_parallel_group()

    parallelism = group.nranks
    if parallelism == 1:
        return input_tensor.clone()

    rank = group.rank
    seq_len = input_tensor.shape[axis]
    if mode == "dual_chunk":
        assert seq_len % (parallelism * 2) == 0, (
            f"Input sequence length {seq_len} can't be divided exactly by "
            f"sequence parallelism * 2 {parallelism * 2}"
        )
        interval = seq_len // parallelism // 2
        chunk_start = paddle.slice(input_tensor, axes=[axis], starts=[interval * rank], ends=[interval * (rank + 1)])
        chunk_end = paddle.slice(
            input_tensor, axes=[axis], starts=[seq_len - interval * (rank + 1)], ends=[seq_len - interval * rank]
        )
        result = paddle.concat([chunk_start, chunk_end], axis=axis)
    elif mode == "balanced_swap":
        assert buckets is not None and len(buckets) == parallelism
        chunk_size = seq_len // sum(len(bucket) for bucket in buckets)
        result = paddle.concat(
            [
                paddle.slice(input_tensor, axes=[axis], starts=[idx * chunk_size], ends=[(idx + 1) * chunk_size])
                for _, idx in buckets[rank]
            ],
            axis=axis,
        )
    else:
        raise ValueError(f"unsupported scatter mode: {mode}")
    # assign copies out so the (large) source tensor can be freed (slice keeps a view)
    return paddle.assign(result)


def all_gather_balance(input_tensor, group=None, axis=0, mode="dual_chunk", buckets=None):
    """Gather CP shares and restore the original sequence order."""
    if group is None:
        group = fleet.get_hybrid_communicate_group().get_context_parallel_group()

    parallelism = group.nranks
    if parallelism == 1:
        return input_tensor.clone()

    if mode == "dual_chunk":
        chunk_start, chunk_end = paddle.split(input_tensor, 2, axis=axis)
        gathered_start_list = [paddle.empty(chunk_start.shape, dtype=input_tensor.dtype) for _ in range(parallelism)]
        dist.stream.all_gather(gathered_start_list, chunk_start, group=group, use_calc_stream=True)
        gathered_end_list = [paddle.empty(chunk_end.shape, dtype=input_tensor.dtype) for _ in range(parallelism)]
        dist.stream.all_gather(gathered_end_list, chunk_end, group=group, use_calc_stream=True)
        gathered_end_list = gathered_end_list[::-1]
        return paddle.concat(gathered_start_list + gathered_end_list, axis=axis)
    if mode != "balanced_swap":
        raise ValueError(f"unsupported gather mode: {mode}")

    assert buckets is not None and len(buckets) == parallelism
    gathered = [paddle.empty(input_tensor.shape, dtype=input_tensor.dtype) for _ in range(parallelism)]
    dist.stream.all_gather(gathered, input_tensor, group=group, use_calc_stream=True)
    chunk_size = input_tensor.shape[axis] // len(buckets[0])
    chunks = [None] * sum(len(bucket) for bucket in buckets)
    for source, bucket in enumerate(buckets):
        for local_pos, (_, original_pos) in enumerate(bucket):
            chunks[int(original_pos)] = paddle.slice(
                gathered[source], axes=[axis], starts=[local_pos * chunk_size], ends=[(local_pos + 1) * chunk_size]
            )
    return paddle.concat(chunks, axis=axis)


def all_gather_contiguous(input_tensor, group=None, axis=0):
    """Gather local tensors in rank order (no reordering)."""
    if group is None:
        group = fleet.get_hybrid_communicate_group().get_context_parallel_group()
    gathered = [paddle.empty(input_tensor.shape, dtype=input_tensor.dtype) for _ in range(group.nranks)]
    dist.stream.all_gather(gathered, input_tensor.contiguous(), group=group, use_calc_stream=True)
    return paddle.concat(gathered, axis=axis)


def reduce_scatter_any_axis_balance(input_tensor, axis, group=None):
    """Balanced reduce-scatter: adjoint of ``all_gather_balance`` (dual-chunk)."""
    if group is None:
        group = fleet.get_hybrid_communicate_group().get_context_parallel_group()

    parallelism = group.nranks
    if parallelism == 1:
        return input_tensor.clone()

    assert input_tensor.shape[axis] % (parallelism * 2) == 0, (
        f"Input sequence length {input_tensor.shape[axis]} can't be "
        f"divided exactly by context parallelism * 2 {parallelism * 2}"
    )

    input_start, input_end = paddle.split(input_tensor, 2, axis=axis)
    chunks_start = paddle.split(input_start, parallelism, axis=axis)
    chunks_end = paddle.split(input_end, parallelism, axis=axis)
    chunks_end = chunks_end[::-1]
    combined_chunks = [
        paddle.concat([s, e], axis=axis) for s, e in zip(chunks_start, chunks_end)
    ]
    output_buffers = [paddle.empty(combined_chunks[0].shape, dtype=input_tensor.dtype) for _ in range(parallelism)]
    dist.stream.alltoall(output_buffers, combined_chunks, group=group, use_calc_stream=True)
    return paddle.stack(output_buffers, axis=0).sum(axis=0)


def reduce_scatter_any_axis_order(input_tensor, axis, group=None):
    """Reduce-scatter a rank-ordered full tensor (adjoint of all_gather_contiguous)."""
    if group is None:
        group = fleet.get_hybrid_communicate_group().get_context_parallel_group()
    chunks = paddle.split(input_tensor, group.nranks, axis=axis)
    buffers = [paddle.empty(chunks[0].shape, dtype=input_tensor.dtype) for _ in range(group.nranks)]
    dist.stream.alltoall(buffers, [chunk.contiguous() for chunk in chunks], group=group, use_calc_stream=True)
    return paddle.stack(buffers).cast("float32").sum(0).cast(input_tensor.dtype)


def preprocess_index_dual_chunks(startend_row_indices, chunk_id_first, chunk_id_second, seq_blocksize, max_seqlen_q):
    """Rebase FlashMask row indices for a rank owning two (front+back) chunks."""
    rows_min_first = chunk_id_first * seq_blocksize
    rows_min_second = chunk_id_second * seq_blocksize

    indices_first = paddle.clip(startend_row_indices - rows_min_first, min=0, max=max_seqlen_q)
    indices_second = paddle.clip(startend_row_indices - rows_min_second, min=0, max=max_seqlen_q)
    indices_second = paddle.where(indices_second != 0, indices_second + max_seqlen_q, indices_second)
    return paddle.maximum(indices_first, indices_second)
