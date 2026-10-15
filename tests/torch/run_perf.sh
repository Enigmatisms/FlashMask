#!/bin/bash
# FlashMask FM-4 benchmark (PyTorch): one head dim per GPU, run in parallel.
#   SM90:  D = 128, 192, 256
#   SM100: D = 128, 192, 256, plus 576 (split and shared KV)
#
# Usage: DEVICES=2,3,4 bash run_perf.sh     (default DEVICES: 0,1,2,...)

cd "$(dirname "$0")"
TS=$(date +%Y%m%d_%H%M%S)

IFS=',' read -r -a GPUS <<< "${DEVICES:-0,1,2,3,4}"
CC_MAJOR=$(nvidia-smi -i "${GPUS[0]}" --query-gpu=compute_cap --format=csv,noheader | cut -d. -f1)

if [ "$CC_MAJOR" -ge 10 ]; then
    HEADDIMS=(128 192 256 576 576)
    KV_MODES=(split split split split shared)
else
    HEADDIMS=(128 192 256)
    KV_MODES=(split split split)
fi

for i in "${!HEADDIMS[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPUS[$i]}" python3 benchmark_flashmask.py \
        --fm_version 4                                                \
        --head_dim "${HEADDIMS[$i]}"                                  \
        --kv_mode "${KV_MODES[$i]}"                                   \
        --dedup_static_masks                                          \
        --current_time "$TS" &
done
wait
