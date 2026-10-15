#!/bin/bash
# FM-4 cp_balance accuracy test launcher (CP4, self-contained sweep).
#
# cp_balance_main.py fixes CP=4 and cycles S_local x head_dim x kv-heads x mask
# in-process, so this launcher takes no shape args. Launch scaffolding mirrors
# run_cp_overlap.sh (PADDLE_* unset dance, NCCL env, TARGET_RANKS node selection,
# hostfile-derived master).
#
# Usage (single node, just run it):
#   bash run_cp_balance.sh
# Run test_cp_balance_launch.py (pytest) for the gated entry point.
#
# Env:
#   TARGET_RANKS  PADDLE_TRAINER_IDs that participate (default "0"); count = nnodes
#   DEVICES       GPUs per node (default: single-node -> 0,1,2,3)
#   PY_BIN        python interpreter (default: python on PATH)
#   MASTER PORT HOSTFILE   rendezvous overrides (single node defaults to localhost)

mpi_rank=${OMPI_COMM_WORLD_RANK:-0}
node_rank=$((mpi_rank+offset))
mpi_node=${OMPI_COMM_WORLD_SIZE:-1}
nnode_train=${nnode_set:-${mpi_node}}
master_train=${master:-localhost}

echo "Distributed Training ${node_rank}/${nnode_train} master=${master_train}"
set -x

unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT
nnodes=$PADDLE_TRAINERS_NUM
rank=$PADDLE_TRAINER_ID

for name in `env | grep -E 'PADDLE|ENDPOINT' | awk -F'=' '{print $1}'`; do
  unset ${name}
done

export NCCL_DEBUG=INFO
unset GLOG_vmodule GLOG_v
export PYTHONUNBUFFERED=1
export FLAGS_use_auto_growth_pinned_allocator=True

export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_IB_TIMEOUT=22
export NCCL_IB_GID_INDEX=3
export NCCL_NVLS_ENABLE=0
export NCCL_IB_ADAPTIVE_ROUTING=1

export PADDLE_PG_TIMEOUT=150000
export CUDA_MODULE_LOADING=LAZY

# release stale shared memory
find /dev/shm/ -type f -name "paddle_*" -print0 | xargs -0 rm -f

# do NOT add /usr/local/cuda/compat here: shim mismatches the driver, zeroes the device count.

# node selection: only the listed PADDLE_TRAINER_IDs participate; count = nnodes.
TARGET_RANKS=(${TARGET_RANKS:-0})

is_target=false
for i in "${!TARGET_RANKS[@]}"; do
    if [[ $rank -eq ${TARGET_RANKS[$i]} ]]; then
        is_target=true
        new_rank=$i
        break
    fi
done

if [[ "$is_target" = false ]]; then
    echo "[run_cp_balance] SKIP: rank=$rank not in TARGET_RANKS=(${TARGET_RANKS[*]}); nothing to run on this node."
    exit 0
fi

rank=$new_rank
nnodes=${#TARGET_RANKS[@]}

first_rank=${TARGET_RANKS[0]}
line_num=$((first_rank + 1))

# master endpoint: single node just uses localhost; multi-node derives it from a
# hostfile. Override with MASTER=<ip> / PORT=<port>, or point HOSTFILE elsewhere.
if [ "$nnodes" -eq 1 ]; then
    master=${MASTER:-localhost}
else
    hostfile=${HOSTFILE:-/root/paddlejob/workspace/hostfile}
    master=${MASTER:-$(cat "$hostfile" | head -n "${line_num}" | tail -n 1 | awk '{print $1}')}
fi
port=${PORT:-12348}

export FLAGS_shard_bypass_dygraph_optimizer=1

# FA-v4 in-library SM100 path + accuracy determinism.
export FLAGS_flash_attn_version=4
export FLAGS_use_cinn=True
export FLAGS_cudnn_deterministic=1
export FLAGS_call_stack_level=2

CP_SIZE=4
PY_BIN=${PY_BIN:-python}

cd "$(dirname "${BASH_SOURCE[0]}")"

# device selection: single node uses GPUs 0..3; multi-node uses all.
if [ -n "${DEVICES:-}" ]; then
    device_arg="--devices ${DEVICES}"
elif [ "$nnodes" -eq 1 ]; then
    device_arg="--devices $(seq -s, 0 $((CP_SIZE - 1)))"
else
    device_arg=""
fi

echo "[run_cp_balance] CP=$CP_SIZE nnodes=$nnodes rank=$rank master=$master:$port"

$PY_BIN -m paddle.distributed.launch \
    --log_dir "log_cp_balance-r$rank" \
    --master "$master:$port" \
    --nnodes "$nnodes" \
    --rank "$rank" \
    --run_mode=collective \
    $device_arg \
    cp_balance_main.py
