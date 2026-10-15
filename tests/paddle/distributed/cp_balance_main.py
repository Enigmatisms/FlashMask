# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""CP4 cp_balance accuracy test against the ordinary DualChunkSwap path.

``flashmask.cp_balance.balance_flashmask_input`` produces a per-rank load-balanced
chunk assignment (``buckets``) and a rebased local mask. This test scatters q/k/v
by that plan, runs the balanced FA-v4 path, and checks its gathered output and
dq/dk/dv against the dual-chunk all-gather reference (compared on rank 0 after
gathering both paths back to the full sequence).

Launched one process per GPU by run_cp_balance.sh (renamed off the ``test_`` prefix
so pytest does not import it directly; the pytest entry point is
test_cp_balance_launch.py).
"""

import sys
from pathlib import Path

import numpy as np
import paddle
from paddle.distributed import fleet

# launched as a script by run_cp_balance.sh, so put the repo root on sys.path for
# the framework-neutral helpers in tests/common (siblings resolve on their own)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from cp_utils import all_gather_balance, scatter_balance
from all_gather_flashmask import flashmask_attention_cp
from balanced_flashmask import balanced_flashmask_attention_fm4
from tests.common.startend_row_indices import DOC_PACKING, GENERATORS
from tests.common.compare import check_accuracy
from flashmask.cp_balance import balance_flashmask_input


CP_SIZE = 4
B = 2
HQ = 16
S_LOCALS = (4096, 8192)
HKS = (1, 4, 16)
HEAD_DIMS = (128, 256)
MASK_TYPES = ("document", "prefix_lm")

strategy = fleet.DistributedStrategy()
strategy.hybrid_configs = {
    "dp_degree": 1,
    "mp_degree": 1,
    "pp_degree": 1,
    "sharding_degree": CP_SIZE,
    "sep_degree": 1,
    "ep_degree": CP_SIZE,
    "moe_sharding_degree": 1,
    "cp_degree": CP_SIZE,
    "order": ["sharding", "moe_sharding", "pp", "sep", "cp", "dp", "ep", "mp"],
}
fleet.init(is_collective=True, strategy=strategy)
cp_group = fleet.get_hybrid_communicate_group().get_context_parallel_group()
rank = paddle.distributed.get_rank()
assert cp_group.world_size == CP_SIZE


def make_mask(mask_type, s_total):
    packing = DOC_PACKING if s_total == 32768 else [(p, d // 2) for p, d in DOC_PACKING]
    arr, _ = GENERATORS[mask_type](B, s_total, s_total, 1, packing)
    return paddle.to_tensor(arr)


def gather_pair(actual, expected, buckets):
    actual = all_gather_balance(
        actual.detach(), group=cp_group, axis=1, mode="balanced_swap", buckets=buckets
    )
    expected = all_gather_balance(expected.detach(), group=cp_group, axis=1, mode="dual_chunk")
    return actual, expected


def check_buckets(buckets, s_total):
    expected = list(range(s_total // 2048))
    indices = [int(idx) for bucket in buckets for _, idx in bucket]
    assert len(buckets) == CP_SIZE
    assert all(len(bucket) == len(expected) // CP_SIZE for bucket in buckets)
    assert len(indices) == len(set(indices)), f"duplicate bucket indices: {indices}"
    assert sorted(indices) == expected, f"bucket indices are not a permutation: {indices}"


def run_case(s_local, hk, d, mask_type, seed):
    s_total = s_local * CP_SIZE
    paddle.seed(seed)
    np.random.seed(seed)
    full_q = paddle.randn([B, s_total, HQ, d], dtype="bfloat16")
    full_k = paddle.randn([B, s_total, hk, d], dtype="bfloat16")
    full_v = paddle.randn([B, s_total, hk, d], dtype="bfloat16")
    full_grad = paddle.randn([B, s_total, HQ, d], dtype="bfloat16")
    mask = make_mask(mask_type, s_total)
    local_mask, buckets = balance_flashmask_input(mask.clone(), CP_SIZE, cp_group.rank)
    check_buckets(buckets, s_total)

    q_ref = scatter_balance(full_q, group=cp_group, axis=1, mode="dual_chunk").contiguous()
    k_ref = scatter_balance(full_k, group=cp_group, axis=1, mode="dual_chunk").contiguous()
    v_ref = scatter_balance(full_v, group=cp_group, axis=1, mode="dual_chunk").contiguous()
    q = scatter_balance(full_q, group=cp_group, axis=1, mode="balanced_swap", buckets=buckets).contiguous()
    k = scatter_balance(full_k, group=cp_group, axis=1, mode="balanced_swap", buckets=buckets).contiguous()
    v = scatter_balance(full_v, group=cp_group, axis=1, mode="balanced_swap", buckets=buckets).contiguous()
    for tensor in (q_ref, k_ref, v_ref, q, k, v):
        tensor.stop_gradient = False

    out_ref = flashmask_attention_cp(q_ref, k_ref, v_ref, mask, group=cp_group)
    out = balanced_flashmask_attention_fm4(q, k, v, local_mask, group=cp_group)
    out_full, out_ref_full = gather_pair(out, out_ref, buckets)
    ok = rank != 0 or check_accuracy(out_full, out_ref_full, "out", rank=rank)

    grad = scatter_balance(
        full_grad, group=cp_group, axis=1, mode="balanced_swap", buckets=buckets
    ).contiguous()
    grad_ref = scatter_balance(
        full_grad, group=cp_group, axis=1, mode="dual_chunk"
    ).contiguous()
    out.backward(grad)
    out_ref.backward(grad_ref)

    for name, actual, expected in (
        ("dq", q.grad, q_ref.grad),
        ("dk", k.grad, k_ref.grad),
        ("dv", v.grad, v_ref.grad),
    ):
        actual_full, expected_full = gather_pair(actual, expected, buckets)
        if rank == 0:
            ok = check_accuracy(actual_full, expected_full, name, rank=rank) and ok
    flag = paddle.to_tensor([int(ok)], dtype="int32")
    paddle.distributed.all_reduce(flag, op=paddle.distributed.ReduceOp.MIN, group=cp_group)
    return bool(flag.item())


def main():
    failed = []
    case_id = 0
    for s_local in S_LOCALS:
        for d in HEAD_DIMS:
            for hk in HKS:
                for mask_type in MASK_TYPES:
                    case_id += 1
                    desc = f"B={B} S_local={s_local} Hq={HQ} Hkv={hk} D={d} mask={mask_type}"
                    if rank == 0:
                        print(f">>> {desc}", flush=True)
                    if not run_case(s_local, hk, d, mask_type, 2024 + case_id):
                        failed.append(desc)
                    if rank == 0:
                        print(f"[{'FAIL' if desc in failed else 'PASS'}] {desc}", flush=True)
    if rank == 0:
        print(f"cp_balance CP4: {case_id - len(failed)}/{case_id} passed", flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
