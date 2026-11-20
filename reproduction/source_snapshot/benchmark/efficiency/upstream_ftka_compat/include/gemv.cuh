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
#ifndef FTKA_GEMV_CUH_
#define FTKA_GEMV_CUH_

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <iostream>

#include "dequantize.cuh"
#include "flashinfer/attention/cascade.cuh"
#include "flashinfer/attention/state.cuh"
#include "flashinfer/cp_async.cuh"
#include "flashinfer/layout.cuh"
#include "flashinfer/math.cuh"
#include "flashinfer/page.cuh"
#include "flashinfer/pos_enc.cuh"
#include "flashinfer/utils.cuh"
#include "flashinfer/vec_dtypes.cuh"
#include "lowbit_dtypes.cuh"

using namespace flashinfer;

namespace ftka {

namespace cg = cooperative_groups;
using cp_async::PrefetchMode;
using cp_async::SharedMemFillMode;

namespace {

// bdx * vec_size = head_dim
// group_size * tile_size_per_bdx = tile_size
template <uint32_t vec_size, uint32_t bdx, uint32_t tile_size, typename DTypeK, typename DTypeQ,
          typename DTypeOut>
__device__ __forceinline__ void compute_qk(const DTypeK* k_smem, const DTypeQ* quant_scales,
                                           const DTypeQ* quant_zeros, vec_t<half, vec_size>& q_vec,
                                           uint32_t kv_idx_base, uint32_t iter_base,
                                           uint32_t iter_bound, DTypeOut* o) {
  uint32_t tx = threadIdx.x, tz = threadIdx.z;
  vec_t<half, vec_size> k_vec;
  half scale = 1, zero = 0;

#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    if constexpr (std::is_same_v<DTypeK, lowbit::uint_4bit>) {
      // k_vec.cast_load(lowbit::get_ptr(k_smem, (j * bdx + tx) * vec_size));
      dequant::fast_dequantize_i4s_to_fp16s<vec_size>(
          lowbit::get_ptr(k_smem, (j * bdx + tx) * vec_size), k_vec, scale, zero);
    } else if constexpr (std::is_same_v<DTypeK, uint8_t>) {
      dequant::fast_dequantize_i8s_to_fp16s<vec_size>(k_smem + (j * bdx + tx) * vec_size, k_vec,
                                                      scale, zero);
      // k_vec.cast_load(k_smem + (j * bdx + tx) * vec_size);
    } else {
      k_vec.cast_load(k_smem + (j * bdx + tx) * vec_size);
    }
    // k_vec.cast_load(k_smem);
    // vec_size: 8 for 16bits, 16 for 8bits, 32 for 4bits
    // at most, fp16*32 = 16 uint32

    float local_sum = 0.f;
    half local_sum1 = 0;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      local_sum1 = __hfma(q_vec[i], k_vec[i], local_sum1);
    }
    local_sum = __half2float(local_sum1);

    //     uint32_t qk;
    //     uint32_t* q_temp = reinterpret_cast<uint32_t*>(q_vec.ptr());
    //     uint32_t* k_temp = reinterpret_cast<uint32_t*>(k_vec.ptr());

    //     asm volatile("mul.f16x2 %0, %1, %2;\n" : "=r"(qk) : "r"(q_temp[0]), "r"(k_temp[0]));
    // #pragma unroll
    //     for (uint32_t i = 1; i < vec_size / 2; i++) {
    //       // local_sum += q_vec[i] * k_vec[i];
    //       asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(qk) : "r"(q_temp[i]),
    //       "r"(k_temp[i]), "r"(qk));
    //     }
    // half2 qk_vec_h = (half2 &)qk;
    // float local_sum = __half2float(__hadd(qk_vec_h.x, qk_vec_h.y));

#pragma unroll
    for (uint32_t offset = bdx / 2; offset > 0; offset /= 2) {
      // local_sum += math::shfl_xor_sync(local_sum, offset);
      local_sum += __shfl_xor_sync(uint32_t(-1), local_sum, offset);
    }
    // Store out
    if (iter_base + tz * tile_size + j < iter_bound) {
      if (tx == 0) {
        o[kv_idx_base + tz * tile_size + j] = static_cast<DTypeOut>(local_sum);
      }
    }
  }
}

}  // namespace

/*!
 * \brief FlashAttention decoding cuda kernel with kv-cache for a single request
 * \tparam pos_encoding_mode The positional encoding mode
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam bdy A template integer indicates the block size in y dimension
 * \tparam DTypeQ A template type indicates the query data type
 * \tparam DTypeK A template type indicates the key-value data type
 * \tparam DTypeO A template type indicates the output data type
 * \param q [num_qo_heads, head_dim] The query matrix
 * \param k [seq_len, num_kv_heads, head_dim] The key matrix in kv-cache
 * \param v [seq_len, num_kv_heads, head_dim] The value matrix in kv-cache
 * \param o [num_qo_heads, head_dim] The output matrix
 * \param head_dim A integer indicates the head dimension
 * \param rope_rcp_scale A floating number indicate the reciprocal
 *   of scaling ratio used in PI(Position Interpolation) for RoPE (Rotary
 *   Positional Embeddings)
 * \param rope_rcp_theta A floating number indicate the reciprocal
 *   of "theta" used in RoPE (Rotary Positional Embeddings)
 * \param kv_chunk_size A integer indicates the kv-chunk size
 */
template <uint32_t num_stages_smem, uint32_t tile_size_per_bdx, uint32_t vec_size, uint32_t bdx,
          uint32_t bdy, uint32_t bdz, typename DTypeQ, typename DTypeK, typename DTypeOut,
          typename IdType>
__global__ void SingleSparseGEMVKernel(DTypeQ* q, DTypeK* k,
                                       // DTypeQ* quant_scales, DTypeQ* quant_zeros,
                                       DTypeOut* o, uint32_t kv_chunk_size, uint32_t num_qo_heads,
                                       uint32_t num_kv_heads, uint32_t seq_len) {
  auto block = cg::this_thread_block();
  auto grid = cg::this_grid();

  constexpr uint32_t head_dim = bdx * vec_size;
  uint32_t kv_head_idx = blockIdx.y;
  uint32_t qo_head_idx = kv_head_idx * bdy + threadIdx.y;
  uint32_t kv_chunk_idx = blockIdx.x;

  extern __shared__ uint8_t smem[];
  DTypeK* k_smem = (DTypeK*)smem;

  uint32_t tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
  vec_t<half, vec_size> q_vec;

  // do not apply rotary embedding to q matrix
  uint32_t q_stride_n = num_qo_heads * head_dim;
  uint32_t q_stride_h = head_dim;
  // NOTE(chaofan): hardcoded NHD
  uint32_t kv_stride_n = num_kv_heads * head_dim;
  uint32_t kv_stride_h = head_dim;

  q_vec.cast_load(q + get_elem_offset_impl(0, qo_head_idx, tx * vec_size, q_stride_n, q_stride_h));
  block.sync();

  uint32_t chunk_start = kv_chunk_idx * kv_chunk_size;
  kv_chunk_size = min(kv_chunk_size, seq_len - chunk_start);
  uint32_t chunk_end = chunk_start + kv_chunk_size;

  // preload k tiles and v tiles
  uint32_t producer_kv_idx_base = chunk_start;
  constexpr uint32_t vec_bits = sizeof(DTypeK) * vec_size * 8;
#pragma unroll
  for (uint32_t iter = 0; iter < num_stages_smem; ++iter) {
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          k_smem + (((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          k + get_elem_offset_impl(producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j,
                                   kv_head_idx, tx * vec_size, kv_stride_n, kv_stride_h),
          producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j < chunk_end);
    }
    cp_async::commit_group();
    producer_kv_idx_base += bdy * bdz * tile_size_per_bdx;
  }

  // pipelining k/v tiles loading and state updating
  uint32_t consumer_kv_idx_base = chunk_start, stage_idx = 0;

#pragma unroll 2
  for (uint32_t iter = 0; iter < ceil_div(kv_chunk_size, tile_size_per_bdx * bdy * bdz); ++iter) {
    // compute qk
    cp_async::wait_group<num_stages_smem - 1>();
    block.sync();

    // NOTE(chaofan): No quant for now
    DTypeQ* quant_scales_ptr = nullptr;
    DTypeQ* quant_zeros_ptr = nullptr;

    compute_qk<vec_size, bdx, bdy * tile_size_per_bdx>(
        k_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim, quant_scales_ptr,
        quant_zeros_ptr, q_vec,
        // chunk_start + iter * tile_size_per_bdx * bdy * bdz,
        consumer_kv_idx_base, iter * tile_size_per_bdx * bdy * bdz, kv_chunk_size,
        // NOTE(chaofan): Write address of o, [1, num_heads, seq_len].
        o + qo_head_idx * seq_len);

    block.sync();
    // load k
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          k_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          k + get_elem_offset_impl(producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j,
                                   kv_head_idx, tx * vec_size, kv_stride_n, kv_stride_h),
          producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j < chunk_end);
    }
    cp_async::commit_group();

    stage_idx = (stage_idx + 1) % num_stages_smem;
    producer_kv_idx_base += tile_size_per_bdx * bdy * bdz;
    consumer_kv_idx_base += tile_size_per_bdx * bdy * bdz;
  }
  cp_async::wait_group<0>();
  block.sync();
}

/*!
 * \brief FlashAttention decoding cuda kernel with paged kv-cache for multiple requests
 * \tparam pos_encoding_mode The positional encoding mode
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam bdy A template integer indicates the block size in y dimension
 * \tparam bdz A template integer indicates the block size in z dimension
 * \tparam DTypeQ A template type indicates the query data type
 * \tparam DTypeK A template type indicates the key-value data type
 * \tparam DTypeO A template type indicates the output data type
 * \tparam IdType A template type indicates the index data type
 * \param q [batch_size, num_qo_heads, head_dim] The query matrix
 * \param paged_kv The paged kv-cache data structure
 * \param quant_scales Share the same layout as kv-cache expect the last dimension (d=1)
 * \param quant_zeros Share the same layout as kv-cache expect the last dimension (d=1)
 */
template <uint32_t num_stages_smem, uint32_t tile_size_per_bdx, uint32_t vec_size, uint32_t bdx,
          uint32_t bdy, uint32_t bdz, typename DTypeQ, typename DTypeK, typename DTypeOut,
          typename IdType>
__global__ void BatchedSparseGEMVKernel(DTypeQ* q, paged_kv_t<DTypeK, IdType> paged_kv,
                                        DTypeQ* quant_scales, DTypeQ* quant_zeros, DTypeOut* o) {
  // static_assert(partition_kv == false && rotary_mode == RotaryMode::kNone);
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

  // Quantization params: scales, zeros
  const bool is_quant = quant_scales != nullptr;

  // paged_kv.last_page_len - 1: hard-code for not considering the last entry
  // which is the last page of original kv-cache.
  // Note that last_page_len should be \in [1, PAGE_SIZE]
  const uint32_t kv_chunk_len = paged_kv.get_length(batch_idx);
  extern __shared__ uint8_t smem[];

  DTypeK* k_smem = (DTypeK*)smem;
  DTypeQ* quant_scales_smem;
  DTypeQ* quant_zeros_smem;
  uint32_t head_dim_byte_size = static_cast<uint32_t>(lowbit::size_of_type<DTypeK>() * head_dim);
  if (is_quant) {
    quant_scales_smem =
        (DTypeQ*)(smem + num_stages_smem * tile_size_per_bdx * bdy * bdz * head_dim_byte_size);
    quant_zeros_smem =
        (DTypeQ*)(smem + num_stages_smem * tile_size_per_bdx * bdy * bdz * head_dim_byte_size +
                  num_stages_smem * tile_size_per_bdx * bdy * bdz * sizeof(DTypeQ));
  }
  size_t* kv_offset_smem =
      (size_t*)(smem + num_stages_smem * tile_size_per_bdx * bdy * bdz * head_dim_byte_size +
                2 * num_stages_smem * tile_size_per_bdx * bdy * bdz * sizeof(DTypeQ));

  const uint32_t tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;

  // No in-kernel RoPE
  // No kv partition
  vec_t<half, vec_size> q_vec;
  q_vec.cast_load(q + (batch_idx * num_qo_heads + qo_head_idx) * head_dim + tx * vec_size);
  block.sync();

  // preload k/v tiles
  uint32_t stage_idx = 0;
  constexpr uint32_t vec_bits =
      static_cast<uint32_t>(lowbit::size_of_type<DTypeK>() * vec_size * 8);
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
// load quant params
//     if (is_quant) {
// #pragma unroll
//       for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
//         quant_scales_smem[(((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j)] =
//             quant_scales[kv_offset[j] / head_dim];
//         quant_zeros_smem[(((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j)] =
//             quant_zeros[kv_offset[j] / head_dim];
//       }
//     }
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          lowbit::get_ptr(k_smem,
                          (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
                              tx * vec_size),
          lowbit::get_ptr(paged_kv.k_data, kv_offset[j]),
          ((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < chunk_size);
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

    auto quant_scales_ptr =
        is_quant ? quant_scales_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx : nullptr;
    auto quant_zeros_ptr =
        is_quant ? quant_zeros_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx : nullptr;

    compute_qk<vec_size, bdx, bdy * tile_size_per_bdx>(
        lowbit::get_ptr(k_smem, (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim),
        quant_scales_ptr, quant_zeros_ptr, q_vec,
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
    // load quant params
    //     if (is_quant) {
    // #pragma unroll
    //       for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
    //         quant_scales_smem[(((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j)] =
    //             quant_scales[kv_offset[j] / head_dim];
    //         quant_zeros_smem[(((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j)] =
    //             quant_zeros[kv_offset[j] / head_dim];
    //       }
    //     }

    // load k tiles
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          lowbit::get_ptr(k_smem,
                          (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
                              tx * vec_size),
          lowbit::get_ptr(paged_kv.k_data, kv_offset[j]),
          (((iter + num_stages_smem) * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < chunk_size);
    }
    cp_async::commit_group();
    stage_idx = (stage_idx + 1) % num_stages_smem;
  }
  cp_async::wait_group<0>();
}

/*!
 * \brief Get the heuristic number of threads per threadblock
 * \param group_size The number of qo heads that maps to the same kv head in GQA.
 * \param sizeof_dtype The size (in terms of bytes) of the input data type
 */
constexpr uint32_t get_heuristic_num_threads(uint32_t group_size, uint32_t sizeof_dtype) {
  if (group_size == 8U) {
    if (sizeof_dtype == 1U) {
      return 256U;  // not enough registers for 512 threads
    } else {
      return 512U;
    }
  } else {
    return 128U;
  }
}

template <uint32_t HEAD_DIM, typename DTypeQ, typename DTypeK, typename DTypeOut, typename IdType>
cudaError_t SingleSparseGEMV(DTypeQ* q, DTypeK* k,
                             // DTypeQ* quant_scales, DTypeQ* quant_zeros,
                             DTypeOut* o, uint32_t num_qo_heads, uint32_t num_kv_heads,
                             uint32_t seq_len, cudaStream_t stream = nullptr) {
  constexpr uint32_t vec_size = std::max(16UL / sizeof(DTypeK), HEAD_DIM / 32UL);
  constexpr uint32_t bdx = HEAD_DIM / vec_size;
  auto compute_capacity = GetCudaComputeCapability();
  static_assert(bdx <= 32U);
  DISPATCH_GQA_GROUP_SIZE(num_qo_heads / num_kv_heads, GROUP_SIZE, {
    constexpr uint32_t bdy = GROUP_SIZE;
    constexpr uint32_t num_threads =
        std::max(get_heuristic_num_threads(GROUP_SIZE, sizeof(DTypeK)), bdx * bdy);
    constexpr uint32_t bdz = num_threads / (bdx * bdy);
    constexpr uint32_t tile_size_per_bdx = GROUP_SIZE == 1 ? (sizeof(DTypeK) == 1 ? 2U : 8U) : 1U;
    DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(compute_capacity, NUM_STAGES_SMEM, {
      const uint32_t smem_size = 1U * NUM_STAGES_SMEM * bdy * tile_size_per_bdx * bdz * HEAD_DIM *
                                 sizeof(DTypeK);  // Just K Cache
      auto kernel = SingleSparseGEMVKernel<NUM_STAGES_SMEM, tile_size_per_bdx, vec_size, bdx, bdy,
                                           bdz, DTypeQ, DTypeK, DTypeOut, IdType>;
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      if (seq_len <= 256) {
        // no need to use partition-kv kernel
        dim3 nblks = dim3(1, num_kv_heads);
        dim3 nthrs = dim3(bdx, bdy, bdz);
        void* args[] = {(void*)&q,
                        (void*)&k,
                        (void*)&o,
                        (void*)&seq_len,
                        (void*)&num_qo_heads,
                        (void*)&num_kv_heads,
                        (void*)&seq_len};
        FLASHINFER_CUDA_CALL(
            cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
      } else {
        // use partition-kv kernel
        int num_blocks_per_sm = 0;
        int num_sm = 0;
        int dev_id = 0;
        FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
        FLASHINFER_CUDA_CALL(
            cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
        FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &num_blocks_per_sm, kernel, num_threads, smem_size));
        uint32_t max_grid_size = uint32_t(num_blocks_per_sm) * uint32_t(num_sm);
        uint32_t max_num_kv_chunks = max_grid_size / num_kv_heads;
        uint32_t kv_chunk_size = max(ceil_div(seq_len, max_num_kv_chunks), 256);
        uint32_t num_chunks = ceil_div(seq_len, kv_chunk_size);
        dim3 nblks = dim3(num_chunks, num_kv_heads);
        if (nblks.x == 0 || nblks.y == 0) {
          std::ostringstream err_msg;
          err_msg << "Invalid kernel configuration: nblks=(" << nblks.x << "," << nblks.y << ")";
          FLASHINFER_ERROR(err_msg.str());
        }
        dim3 nthrs = dim3(bdx, bdy, bdz);
        void* args[] = {(void*)&q,
                        (void*)&k,
                        (void*)&o,
                        (void*)&kv_chunk_size,
                        (void*)&num_qo_heads,
                        (void*)&num_kv_heads,
                        (void*)&seq_len};
        FLASHINFER_CUDA_CALL(
            cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
      }
    });
  });
  return cudaSuccess;
}

template <uint32_t HEAD_DIM, typename DTypeQ, typename DTypeK, typename DTypeOut, typename IdType>
cudaError_t BatchedSparseGEMV(DTypeQ* q, paged_kv_t<DTypeK, IdType> paged_kv, DTypeQ* scales,
                              DTypeQ* zeros, DTypeOut* o, uint32_t num_qo_heads,
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

  constexpr uint32_t vec_size =
      std::max(static_cast<size_t>(16 / lowbit::size_of_type<DTypeK>()), HEAD_DIM / 32UL);
  auto compute_capacity = GetCudaComputeCapability();

  constexpr uint32_t bdx = HEAD_DIM / vec_size;
  static_assert(bdx <= 32);
  DISPATCH_GQA_GROUP_SIZE(num_qo_heads / num_kv_heads, GROUP_SIZE, {
    constexpr uint32_t bdy = GROUP_SIZE;
    constexpr uint32_t num_threads = std::max(128U, bdx * bdy);
    constexpr uint32_t num_stages_smem = 2U;
    constexpr uint32_t bdz = num_threads / (bdx * bdy);
    constexpr uint32_t tile_size_per_bdx =
        GROUP_SIZE == 1 ? (lowbit::size_of_type<DTypeK>() <= 1 ? 2U : 4U) : 1U;
    DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(compute_capacity, NUM_STAGES_SMEM, {
      const uint32_t smem_size =
          NUM_STAGES_SMEM * tile_size_per_bdx * bdy * bdz *
              static_cast<uint32_t>(lowbit::size_of_type<DTypeK>() * HEAD_DIM) +
          2 * NUM_STAGES_SMEM * tile_size_per_bdx * bdy * bdz * sizeof(DTypeQ) +  // quant params
          std::max(tile_size_per_bdx * num_threads * sizeof(DTypeK*),
                   2 * bdy * bdz * sizeof(float));  // kv_offsets
      dim3 nblks(batch_size, num_kv_heads);
      dim3 nthrs(bdx, bdy, bdz);
      // std::cout << "BatchedSparseGEMV launch config: nblks=(" << nblks.x << "," << nblks.y
      //           << "), nthrs=(" << nthrs.x << "," << nthrs.y << "," << nthrs.z
      //           << "), smem_size=" << smem_size << std::endl;
      auto kernel = BatchedSparseGEMVKernel<num_stages_smem, tile_size_per_bdx, vec_size, bdx, bdy,
                                            bdz, DTypeQ, DTypeK, DTypeOut, IdType>;
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      void* args[] = {(void*)&q, (void*)&paged_kv, (void*)&scales, (void*)&zeros, (void*)&o};
      FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
    })
  });
  return cudaSuccess;
}

}  // namespace ftka

#endif  // FTKA_GEMV_CUH_
