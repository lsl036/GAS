/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
  This file is modified based on URL:
      https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/decode.cuh
  Support for Page-Sparsity Self-Attention by dynamic selection.
*/

#ifndef FTKA_QUEST_GEMV_CUH_
#define FTKA_QUEST_GEMV_CUH_

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#ifdef FLASHINFER_ENABLE_FP8
#include <cuda_fp8.h>
#endif
#include <cuda_runtime.h>

#include <cuda/pipeline>
#include <iostream>
#include <random>

#include "flashinfer/attention/cascade.cuh"
#include "flashinfer/attention/state.cuh"
#include "flashinfer/cp_async.cuh"
#include "flashinfer/math.cuh"
#include "flashinfer/page.cuh"
#include "flashinfer/pos_enc.cuh"
#include "flashinfer/utils.cuh"
#include "flashinfer/vec_dtypes.cuh"

using namespace flashinfer;

namespace ftka {

namespace cg = cooperative_groups;
using cp_async::PrefetchMode;
using cp_async::SharedMemFillMode;

namespace {

/*!
 * \brief Load (max, min) tile of K from smem and compute accumulated max qk
 * \tparam head_dim A template integer indicates the head dimension
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam tile_size A template integer indicates the tile size per (bdx * bdy) threads.
 * \tparam T A template type indicates the input data type
 * \param max_smem A pointer to the start of max smem, bind to bdz
 * \param min_smem A pointer to the start of min smem, bind to bdz
 * \param q_vec A vector of float indicates the thread-local query vector
 * \param kv_idx_base A integer indicates the block-local kv position in kv-cache
 * \param iter_base A integer indicates the block-local iteration index
 * \param iter_bound A integer indicates the block-local iteration bound
 * \param o A pointer to the start of global memory within this specific head
 */
template <uint32_t vec_size, uint32_t bdx, uint32_t tile_size, typename DTypeK, typename DTypeOut>
__device__ __forceinline__ void compute_max_possible(const DTypeK* max_smem, const DTypeK* min_smem,
                                                     const vec_t<float, vec_size>& q_vec,
                                                     uint32_t kv_idx_base, uint32_t iter_base,
                                                     uint32_t iter_bound, DTypeOut* o) {
  uint32_t tx = threadIdx.x, tz = threadIdx.z;
#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    vec_t<float, vec_size> max_vec;
    vec_t<float, vec_size> min_vec;
    max_vec.cast_load(max_smem + (j * bdx + tx) * vec_size);
    min_vec.cast_load(min_smem + (j * bdx + tx) * vec_size);
    float max_possible = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      max_possible += max(q_vec[i] * max_vec[i], q_vec[i] * min_vec[i]);
    }
#pragma unroll
    for (uint32_t offset = bdx / 2; offset > 0; offset /= 2) {
      max_possible += math::shfl_xor_sync(max_possible, offset);
    }
    // Store out
    if (iter_base + tz * tile_size + j < iter_bound) {
      if (tx == 0) {
        o[kv_idx_base + tz * tile_size + j] = static_cast<DTypeOut>(max_possible);
      }
    }
  }
}

}  // namespace

template <uint32_t num_stages_smem, uint32_t tile_size_per_bdx, uint32_t vec_size, uint32_t bdx,
          uint32_t bdy, uint32_t bdz, typename DTypeQ, typename DTypeK, typename DTypeOut,
          typename IdType>
__global__ void MaxPossibleSampleWithPagedKVCacheKernel(DTypeQ* __restrict__ q,
                                                        paged_kv_t<DTypeK, IdType> paged_kv,
                                                        DTypeOut* __restrict__ o) {
  auto block = cg::this_thread_block();

  constexpr uint32_t head_dim = bdx * vec_size;
  const uint32_t batch_idx = blockIdx.x;
  const uint32_t kv_head_idx = blockIdx.y;
  const uint32_t qo_head_idx = kv_head_idx * bdy + threadIdx.y;
  const uint32_t num_qo_heads = gridDim.y * bdy;

  const uint32_t kv_len = paged_kv.get_length(batch_idx);

  // NOTE(chaofan):
  //  For now, we donnot use partition kv in batch case.
  //  For futher extension, we can introduce FlashInfer's load balancing (plan) algo in GEMV.
  const uint32_t max_chunk_size = kv_len;
  const uint32_t chunk_start = 0;
  const uint32_t chunk_end = kv_len;
  const uint32_t chunk_size = chunk_end - chunk_start;

  // paged_kv.last_page_len - 1: hard-code for not considering the last entry
  // which is the last page of original kv-cache.
  // Note that last_page_len should be \in [1, PAGE_SIZE]
  const uint32_t kv_chunk_len = paged_kv.get_length(batch_idx);
  extern __shared__ uint8_t smem[];

  // Maximum value and Minimum share the same layout as k and v
  // Therefore we denote them by K and V
  DTypeK* k_smem = (DTypeK*)smem;
  DTypeK* v_smem =
      (DTypeK*)(smem + num_stages_smem * tile_size_per_bdx * bdy * bdz * head_dim * sizeof(DTypeK));
  size_t* kv_offset_smem = (size_t*)(smem + 2 * num_stages_smem * tile_size_per_bdx * bdy * bdz *
                                                head_dim * sizeof(DTypeK));

  const uint32_t tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;

  // No in-kernel RoPE
  // No kv partition
  vec_t<float, vec_size> q_vec;
  q_vec.cast_load(q + (batch_idx * num_qo_heads + qo_head_idx) * head_dim + tx * vec_size);
  block.sync();

  // preload k/v tiles
  uint32_t stage_idx = 0;
  constexpr uint32_t vec_bits = sizeof(DTypeK) * vec_size * 8;
  static_assert(vec_bits == 128, "We want to highly utilize the memory bandwidth");
  const IdType last_indptr = paged_kv.indptr[paged_kv.batch_size];

  static_assert(num_stages_smem <= bdx);
  uint32_t packed_page_iter_base = paged_kv.indptr[batch_idx] * paged_kv.page_size + chunk_start;
#pragma unroll
  for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
    uint32_t q, r;
    paged_kv.page_size.divmod(packed_page_iter_base + ((j * bdz + tz) * bdy + ty) * bdx + tx, q, r);
    kv_offset_smem[((j * bdz + tz) * bdy + ty) * bdx + tx] =
        paged_kv.protective_get_kv_offset(q, kv_head_idx, r, 0, last_indptr);
  }
  block.sync();

  size_t kv_offset[tile_size_per_bdx];
#pragma unroll
  for (uint32_t iter = 0; iter < num_stages_smem; ++iter) {
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      kv_offset[j] =
          kv_offset_smem[((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j] + tx * vec_size;
    }
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          k_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          paged_kv.k_data + kv_offset[j],
          ((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < kv_chunk_len);
    }
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
          v_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          paged_kv.v_data + kv_offset[j],
          ((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < kv_chunk_len);
    }
    cp_async::commit_group();
    stage_idx = (stage_idx + 1) % num_stages_smem;
  }

#pragma unroll 2
  for (uint32_t iter = 0; iter < ceil_div(kv_chunk_len, tile_size_per_bdx * bdy * bdz); ++iter) {
    if ((iter + num_stages_smem) % bdx == 0) {
#pragma unroll
      for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
        uint32_t q, r;
        paged_kv.page_size.divmod(
            packed_page_iter_base + ((iter + num_stages_smem) * tile_size_per_bdx * bdy * bdz +
                                     ((j * bdz + tz) * bdy + ty) * bdx + tx),
            q, r);
        kv_offset_smem[((j * bdz + tz) * bdy + ty) * bdx + tx] =
            paged_kv.protective_get_kv_offset(q, kv_head_idx, r, 0, last_indptr);
      }
    }
    // compute qk
    cp_async::wait_group<num_stages_smem - 1>();
    block.sync();

    compute_max_possible<vec_size, bdx, bdy * tile_size_per_bdx>(
        k_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim,
        v_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim, q_vec,
        chunk_start + iter * tile_size_per_bdx * bdy * bdz, iter * tile_size_per_bdx * bdy * bdz,
        chunk_size,
        // NOTE(chaofan): Write address of o
        o + (num_qo_heads * batch_idx + qo_head_idx) * kv_chunk_len);
    block.sync();

#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      kv_offset[j] = kv_offset_smem[((((iter + num_stages_smem) % bdx) * bdz + tz) * bdy + ty) *
                                        tile_size_per_bdx +
                                    j] +
                     tx * vec_size;
    }
    // load k tiles
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          k_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          paged_kv.k_data + kv_offset[j],
          (((iter + num_stages_smem) * bdz + tz) * bdy + ty) * tile_size_per_bdx + j <
              kv_chunk_len);
    }
    // load v tiles
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
          v_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          paged_kv.v_data + kv_offset[j],
          (((iter + num_stages_smem) * bdz + tz) * bdy + ty) * tile_size_per_bdx + j <
              kv_chunk_len);
    }
    cp_async::commit_group();
    stage_idx = (stage_idx + 1) % num_stages_smem;
  }
  cp_async::wait_group<0>();
}

/*!
 * \brief FlashAttention decoding cuda kernel with paged kv-cache for single request
 * \tparam page_storage Whether to store indices or pointers of each active page
 * \tparam DTypeQ A template type indicates the input data type
 * \tparam DTypeOut A template type indicates the output data type
 * \tparam IdType A template type indicates the index data type used in paged kv-cache
 * \param q [num_qo_heads, head_dim] The query matrix, the layout is fixed NHD
 * \param paged_kv The paged kv cache data structure
 * \param o [num_qo_heads, output_len] The output matrix. Note that output_len = elements in
 * page_kv_t - 1. We ignore the last element because later we manually append it to the selected
 * topk. \param num_qo_heads A integer indicates the number of heads of query and output \param
 * stream The cuda stream to launch the kernel \return status Indicates whether CUDA calls are
 * successful
 */
template <uint32_t HEAD_DIM, typename DTypeQ, typename DTypeK, typename DTypeOut, typename IdType>
cudaError_t MaxPossibleSampleWithPagedKVCache(DTypeQ* q, paged_kv_t<DTypeK, IdType> paged_kv,
                                              DTypeOut* o, uint32_t num_qo_heads,
                                              cudaStream_t stream = nullptr) {
  const uint32_t num_kv_heads = paged_kv.num_heads;
  const uint32_t head_dim = paged_kv.head_dim;
  const uint32_t batch_size = paged_kv.batch_size;
  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads " << num_qo_heads << " is not a multiple of num_kv_heads "
            << num_kv_heads;
    throw std::invalid_argument(err_msg.str());
  }

  DISPATCH_GQA_GROUP_SIZE(num_qo_heads / num_kv_heads, GROUP_SIZE, {
    constexpr uint32_t vec_size = std::max(16UL / sizeof(DTypeQ), HEAD_DIM / 32UL);
    constexpr uint32_t num_stages_smem = 2U;
    constexpr uint32_t bdx = HEAD_DIM / vec_size;
    static_assert(bdx <= 32);
    constexpr uint32_t bdy = GROUP_SIZE;
    constexpr uint32_t num_threads = std::max(128U, bdx * bdy);
    constexpr uint32_t bdz = num_threads / (bdx * bdy);
    constexpr uint32_t tile_size_per_bdx = GROUP_SIZE == 1 ? (sizeof(DTypeQ) == 1 ? 2U : 4U) : 1U;
    const uint32_t smem_size =
        2 * num_stages_smem * tile_size_per_bdx * bdy * bdz * head_dim * sizeof(DTypeK) +
        std::max(tile_size_per_bdx * num_threads * sizeof(DTypeK*),
                 2 * bdy * bdz * sizeof(float));  // kv_offsets
    dim3 nblks(batch_size, num_kv_heads);
    dim3 nthrs(bdx, bdy, bdz);
    auto kernel =
        MaxPossibleSampleWithPagedKVCacheKernel<num_stages_smem, tile_size_per_bdx, vec_size, bdx,
                                                bdy, bdz, DTypeQ, DTypeK, DTypeOut, IdType>;
    FLASHINFER_CUDA_CALL(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    void* args[] = {(void*)&q, (void*)&paged_kv, (void*)&o};
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
  });
  return cudaSuccess;
}

}  // namespace ftka

#endif  // FTKA_QUEST_GEMV_CUH_
