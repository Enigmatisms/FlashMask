// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 The FlashMask Authors. All rights reserved.

#include <climits>
#include <cstdint>
#include <cuda_runtime.h>

#include "cp_balance_fast.hpp"

namespace {

inline cudaStream_t as_stream(uint64_t handle) {
    return reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(handle));
}

inline const int* as_cptr(uint64_t p) {
    return reinterpret_cast<const int*>(static_cast<uintptr_t>(p));
}

inline int* as_ptr(uint64_t p) {
    return reinterpret_cast<int*>(static_cast<uintptr_t>(p));
}

template<int kBlockN>
__global__
void scanMaxMinChunkedKernel(
    const int *input, int b, int n, int *maxo, int *mino) {
    int bid = threadIdx.y + blockIdx.y * blockDim.y;
    if (bid >= b) return;
    int i_offset = bid * n;
    input = input + i_offset;
    const int nblock_seqlen = ((n + kBlockN - 1) / kBlockN + 3) & 0xfffffffc;
    constexpr int nums = (kBlockN + 31) / 32;
    int warpId = blockIdx.x;
    int tid = threadIdx.x;
    int lane_id = threadIdx.x % 32;
    int maxv, minv;
    int idx = warpId * kBlockN + tid;
    if (warpId * kBlockN + kBlockN > n) {
        maxv = 0;
        minv = INT_MAX;
        #pragma unroll
        for (int i = 0; i < nums; i++) {
            if (idx < n && lane_id + i * 32 < kBlockN) {
                maxv = max(maxv, input[idx]);
                minv = min(minv, input[idx]);
            }
            idx += 32;
        }
    } else {
        maxv = 0;
        minv = INT_MAX;
        #pragma unroll
        for (int i = 0; i < nums; i++) {
            if(lane_id + i * 32 < kBlockN) {
                maxv = max(maxv, input[idx]);
                minv = min(minv, input[idx]);
                idx += 32;
            }
        }
    }
    __syncwarp();
    maxv = __reduce_max_sync(0xffffffff, maxv);
    minv = __reduce_min_sync(0xffffffff, minv);
    if (tid == 0) {
        maxo[bid * nblock_seqlen + warpId] = maxv;
        mino[bid * nblock_seqlen + warpId] = minv;
    }
}

// Enum for pointer dispatching in reduce_workload_kernel
enum PtrDispatch { SINGLE_PTR = 1, DUAL_PTR = 2, FULL_PTR = 4 };

template<int kBlockM, int PTR_DISPATCH_TAG, bool is_causal>
__global__ void reduce_workload_kernel(
    const int* LTStartMax, const int* LTStartMin,
    const int* LTEndMax, const int* LTEndMin,
    const int* UTStartMax, const int* UTStartMin,
    const int* UTEndMax, const int* UTEndMin,
    int* workload, // [B, H, Tr, 1]
    int BH, int Tr, int Tc, int S,
    int Br // m_block_size
) {
    const int bh = blockIdx.y;
    const int tr = blockIdx.x;
    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;

    if (tr >= Tr) return;

    // m_block_s/e: Q block boundaries within a single (batch, head).
    const int m_block_s = tr * kBlockM;
    const int m_block_e = m_block_s + kBlockM < S ? m_block_s + kBlockM : S;

    const int bh_offset = bh * Tc;
    const int q_idx = bh * Tr + tr;

    // Stride loop: 每个 thread 处理 tc = threadIdx.x, threadIdx.x + blockDim.x, ...
    int thread_wl = 0;
    for (int tc = static_cast<int>(threadIdx.x); tc < Tc; tc += static_cast<int>(blockDim.x)) {
        const int idx = bh_offset + tc;

        int lt_start_max_val = LTStartMax[idx];
        bool fully_masked = true;

        if constexpr (PTR_DISPATCH_TAG == FULL_PTR) {
            int lt_end_min_val  = LTEndMin[idx];
            int ut_start_max_val = UTStartMax[idx];
            int ut_end_min_val  = UTEndMin[idx];
            fully_masked = (m_block_s >= lt_start_max_val && m_block_e <= lt_end_min_val) ||
                           (m_block_s >= ut_start_max_val && m_block_e <= ut_end_min_val);
        }
        else if constexpr (PTR_DISPATCH_TAG == DUAL_PTR) {
            if constexpr (is_causal) {
                int lt_end_min_val = LTEndMin[idx];
                fully_masked = m_block_s >= lt_start_max_val && m_block_e <= lt_end_min_val;
            } else {
                int ut_end_min_val = UTEndMin[idx];
                fully_masked = (m_block_s >= lt_start_max_val) || (m_block_e <= ut_end_min_val);
            }
        }
        else if constexpr (PTR_DISPATCH_TAG == SINGLE_PTR) {
            fully_masked = m_block_s >= lt_start_max_val;
        }

        thread_wl += fully_masked ? 0 : 1;
    }

    // Warp reduce sum
    __shared__ int smem[32];
    const unsigned mask = 0xffffffff;
    int wl_sum = thread_wl;
    for (int offset = 16; offset > 0; offset >>= 1) {
        wl_sum += __shfl_down_sync(mask, wl_sum, offset);
    }
    if (laneId == 0) {
        smem[warpId] = wl_sum;
    }
    __syncthreads();

    // Final reduce across warps (first warp collects)
    if (threadIdx.x < 32) {
        int val = (threadIdx.x < (blockDim.x + 31) / 32) ? smem[threadIdx.x] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(mask, val, offset);
        }
        if (threadIdx.x == 0) {
            workload[q_idx] = val;
        }
    }
}

__global__ void indices_to_chunks_kernel(
    const int* startend_row_indices,
    const int* chunk_bucket_indices,
    int* chunked_result,
    int num_rows,
    int num_buckets,
    int chunk_size)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    int max_chunk_index = 0;
    int row_val = startend_row_indices[row];

    for (int bucket = 0; bucket < num_buckets; ++bucket) {
        int bucket_idx = chunk_bucket_indices[bucket];
        int chunk_start = bucket_idx * chunk_size;
        int local_index = row_val - chunk_start;
        local_index = max(local_index, 0);
        local_index = min(local_index, chunk_size);

        if (local_index > 0) {
            local_index += bucket * chunk_size;
        }

        if (bucket == 0 || local_index > max_chunk_index) {
            max_chunk_index = local_index;
        }
    }
    chunked_result[row] = max_chunk_index;
}

__global__ void indices_rerank_kernel(
    const int* startend_row_indices,
    int* output_reranked_indices,
    const int* chunk_indices,
    int batch_size,
    int num_heads,
    int seq_len,
    int feature_dim,
    int num_chunks,
    int chunk_size
) {
    int output_seq_len = num_chunks * chunk_size;
    int total_elements = batch_size * output_seq_len * num_heads * feature_dim;
    int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat_idx >= total_elements) return;

    int d = flat_idx % feature_dim;
    int s_out = (flat_idx / feature_dim) % output_seq_len;
    int h = (flat_idx / feature_dim / output_seq_len) % num_heads;
    int b = (flat_idx / feature_dim / output_seq_len / num_heads) % batch_size;

    int chunk_id = s_out / chunk_size;
    int chunk_offset = s_out % chunk_size;
    int src_s = chunk_indices[chunk_id] * chunk_size + chunk_offset;

    if (src_s >= seq_len) return;

    int src_flat_idx = ((b * num_heads + h) * seq_len + src_s) * feature_dim + d;
    int dst_flat_idx = flat_idx;

    output_reranked_indices[dst_flat_idx] = startend_row_indices[src_flat_idx];
}

}  // namespace

namespace {

template <int kBlockM>
void launch_reduce_workload(
    const int* lt_start_max, const int* lt_start_min,
    const int* lt_end_max, const int* lt_end_min,
    const int* ut_start_max, const int* ut_start_min,
    const int* ut_end_max, const int* ut_end_min,
    int* workload, int batch_times_heads, int num_row_blocks,
    int Tc, int stride, int row_block_size, bool is_causal, cudaStream_t stream) {
    dim3 block_dim(1024, 1);
    dim3 grid_dim(num_row_blocks, batch_times_heads);

    int ptr_dispatch_tag = SINGLE_PTR;
    if (lt_end_max || ut_end_max) {
        ptr_dispatch_tag = DUAL_PTR;
        if (ut_start_max) {
            ptr_dispatch_tag = FULL_PTR;
        }
    }

    if (ptr_dispatch_tag == FULL_PTR) {
        reduce_workload_kernel<kBlockM, FULL_PTR, false><<<grid_dim, block_dim, 0, stream>>>(
            lt_start_max, lt_start_min, lt_end_max, lt_end_min,
            ut_start_max, ut_start_min, ut_end_max, ut_end_min,
            workload, batch_times_heads, num_row_blocks, Tc, stride, row_block_size);
    } else if (ptr_dispatch_tag == DUAL_PTR) {
        if (is_causal) {
            reduce_workload_kernel<kBlockM, DUAL_PTR, true><<<grid_dim, block_dim, 0, stream>>>(
                lt_start_max, lt_start_min, lt_end_max, lt_end_min,
                ut_start_max, ut_start_min, ut_end_max, ut_end_min,
                workload, batch_times_heads, num_row_blocks, Tc, stride, row_block_size);
        } else {
            reduce_workload_kernel<kBlockM, DUAL_PTR, false><<<grid_dim, block_dim, 0, stream>>>(
                lt_start_max, lt_start_min, lt_end_max, lt_end_min,
                ut_start_max, ut_start_min, ut_end_max, ut_end_min,
                workload, batch_times_heads, num_row_blocks, Tc, stride, row_block_size);
        }
    } else {
        reduce_workload_kernel<kBlockM, SINGLE_PTR, false><<<grid_dim, block_dim, 0, stream>>>(
            lt_start_max, lt_start_min, lt_end_max, lt_end_min,
            ut_start_max, ut_start_min, ut_end_max, ut_end_min,
            workload, batch_times_heads, num_row_blocks, Tc, stride, row_block_size);
    }
}

}  // namespace

extern "C" {

// input [total_batch, num_sequences] int32; outputs max/min are pre-allocated
// [total_batch, num_blocks_seqlen] where num_blocks_seqlen is padded to a
// multiple of 4 (computed identically on the Python side).
void fm_cpb_scan_max_min(uint64_t input, int total_batch, int num_sequences,
                         int block_size_n, uint64_t maxo, uint64_t mino,
                         uint64_t stream) noexcept {
    dim3 block_dim(32, 4);
    dim3 grid_dim((num_sequences + block_size_n - 1) / block_size_n,
                  (total_batch + 3) / 4);
    const int* in = as_cptr(input);
    int* mx = as_ptr(maxo);
    int* mn = as_ptr(mino);
    cudaStream_t s = as_stream(stream);
    switch (block_size_n) {
        case 32:
            scanMaxMinChunkedKernel<32><<<grid_dim, block_dim, 0, s>>>(in, total_batch, num_sequences, mx, mn);
            break;
        case 64:
            scanMaxMinChunkedKernel<64><<<grid_dim, block_dim, 0, s>>>(in, total_batch, num_sequences, mx, mn);
            break;
        case 96:
            scanMaxMinChunkedKernel<96><<<grid_dim, block_dim, 0, s>>>(in, total_batch, num_sequences, mx, mn);
            break;
        case 128:
            scanMaxMinChunkedKernel<128><<<grid_dim, block_dim, 0, s>>>(in, total_batch, num_sequences, mx, mn);
            break;
        default:
            break;
    }
}

// 8 index tensors (0 == absent); reduce to workload [batch_times_heads, num_row_blocks].
// Tc is the padded per-(bh) column stride (= scan output's last dim).
void fm_cpb_reduce_workload(uint64_t lt_start_max, uint64_t lt_start_min,
                            uint64_t lt_end_max, uint64_t lt_end_min,
                            uint64_t ut_start_max, uint64_t ut_start_min,
                            uint64_t ut_end_max, uint64_t ut_end_min,
                            uint64_t workload, int batch_times_heads,
                            int num_row_blocks, int Tc, int stride,
                            int row_block_size, int is_causal, int m_block_size,
                            uint64_t stream) noexcept {
    cudaStream_t s = as_stream(stream);
    switch (m_block_size) {
        case 64:
            launch_reduce_workload<64>(as_cptr(lt_start_max), as_cptr(lt_start_min),
                as_cptr(lt_end_max), as_cptr(lt_end_min), as_cptr(ut_start_max),
                as_cptr(ut_start_min), as_cptr(ut_end_max), as_cptr(ut_end_min),
                as_ptr(workload), batch_times_heads, num_row_blocks, Tc, stride,
                row_block_size, is_causal != 0, s);
            break;
        case 96:
            launch_reduce_workload<96>(as_cptr(lt_start_max), as_cptr(lt_start_min),
                as_cptr(lt_end_max), as_cptr(lt_end_min), as_cptr(ut_start_max),
                as_cptr(ut_start_min), as_cptr(ut_end_max), as_cptr(ut_end_min),
                as_ptr(workload), batch_times_heads, num_row_blocks, Tc, stride,
                row_block_size, is_causal != 0, s);
            break;
        case 128:
            launch_reduce_workload<128>(as_cptr(lt_start_max), as_cptr(lt_start_min),
                as_cptr(lt_end_max), as_cptr(lt_end_min), as_cptr(ut_start_max),
                as_cptr(ut_start_min), as_cptr(ut_end_max), as_cptr(ut_end_min),
                as_ptr(workload), batch_times_heads, num_row_blocks, Tc, stride,
                row_block_size, is_causal != 0, s);
            break;
        default:
            break;
    }
}

void fm_cpb_indices_to_chunks(uint64_t row_indices, uint64_t chunk_bucket_indices,
                              uint64_t chunked_result, int num_rows, int num_buckets,
                              int chunk_size, uint64_t stream) noexcept {
    const int threads = 256;
    const int blocks = (num_rows + threads - 1) / threads;
    indices_to_chunks_kernel<<<blocks, threads, 0, as_stream(stream)>>>(
        as_cptr(row_indices), as_cptr(chunk_bucket_indices), as_ptr(chunked_result),
        num_rows, num_buckets, chunk_size);
}

void fm_cpb_indices_rerank(uint64_t input_row_indices, uint64_t chunk_indices,
                           uint64_t reranked, int batch_size, int num_heads,
                           int seq_len, int feature_dim, int num_chunks,
                           int chunk_size, uint64_t stream) noexcept {
    const int output_seq_len = num_chunks * chunk_size;
    const int total = batch_size * output_seq_len * num_heads * feature_dim;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    indices_rerank_kernel<<<blocks, threads, 0, as_stream(stream)>>>(
        as_cptr(input_row_indices), as_ptr(reranked), as_cptr(chunk_indices),
        batch_size, num_heads, seq_len, feature_dim, num_chunks, chunk_size);
}

// CPU LPT/IPO solver (host pointers). N items, M workers, writes assign_out[M*K]
// (K = N/M) and returns the minimised maximum worker load.
int fm_cpb_ipo_solve(uint64_t weights, int N, int M, uint64_t assign_out) noexcept {
    return CpBalanceSolver::solve_to(
        reinterpret_cast<const int*>(static_cast<uintptr_t>(weights)),
        N, M, reinterpret_cast<int*>(static_cast<uintptr_t>(assign_out)));
}

}  // extern "C"


