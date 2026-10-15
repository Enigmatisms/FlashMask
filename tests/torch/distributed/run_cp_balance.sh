#!/bin/bash
# FM-4 cp_balance accuracy test launcher (torch, CP4, self-contained sweep).
#
# cp_balance_main.py fixes CP=4 and cycles S_local x head_dim x kv-heads x mask
# in-process, so this launcher takes no shape args. Single-node torchrun,
# mirroring run_cp_overlap.sh.
#
# Usage (single node, just run it):
#   bash run_cp_balance.sh
# Run test_cp_balance_launch.py (pytest) for the gated entry point.
#
# Env:
#   DEVICES   GPUs to use (default: first 4, i.e. 0,1,2,3)
#   PY_BIN    python interpreter (default: python on PATH)
#   PORT      torchrun rendezvous port (default 29500)
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

export PYTHONUNBUFFERED=1
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_IB_TIMEOUT=22
export NCCL_IB_GID_INDEX=3
export NCCL_NVLS_ENABLE=0
export NCCL_IB_ADAPTIVE_ROUTING=1
export CUDA_MODULE_LOADING=LAZY

# do NOT add /usr/local/cuda/compat here: shim mismatches the driver, zeroes the device count.

CP_SIZE=4
PY_BIN=${PY_BIN:-python}
PORT=${PORT:-29500}

# device selection: default the first CP_SIZE GPUs.
if [ -n "${DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES="${DEVICES}"
else
    export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((CP_SIZE - 1)))"
fi

export FLAGS_flash_attn_version=4

logdir="log_cp_balance"
echo "[run_cp_balance] CP=$CP_SIZE devices=$CUDA_VISIBLE_DEVICES port=$PORT"

$PY_BIN -m torch.distributed.run \
    --nnodes 1 \
    --nproc_per_node "$CP_SIZE" \
    --rdzv_backend c10d \
    --rdzv_endpoint "localhost:$PORT" \
    --log_dir "$logdir" \
    --redirects 3 --tee 3 \
    cp_balance_main.py
