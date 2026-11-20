/*
 * Copyright (c) 2024 by FlashInfer team.
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

#include "gemv.cuh"
#include "pytorch_extension_utils.h"
#include "quest_gemv.cuh"
#include "raft_topk.cuh"
#include <ATen/cuda/CUDAContext.h>

using namespace flashinfer;
using namespace ftka;

template <typename DTypeQ>
__global__ void SimpleBatchedInt4GEMVKernel(DTypeQ* q, DTypeQ* o, uint8_t* k_cache,
                                           DTypeQ* quant_scales, DTypeQ* quant_zeros,
                                           int32_t* paged_kv_indices, int32_t* paged_kv_indptr,
                                           int32_t* paged_kv_last_page_len, int32_t batch_size,
                                           int32_t num_heads, int32_t head_dim,
                                           int32_t page_size, int32_t kv_len) {
  const int32_t token_idx = blockIdx.x;
  const int32_t head_idx = blockIdx.y;
  const int32_t batch_idx = blockIdx.z;
  const int32_t tid = threadIdx.x;

  if (batch_idx >= batch_size || head_idx >= num_heads || token_idx >= kv_len) {
    return;
  }

  const int32_t request_pages = paged_kv_indptr[batch_idx + 1] - paged_kv_indptr[batch_idx];
  if (request_pages <= 0) {
    return;
  }
  const int32_t valid_len = (request_pages - 1) * page_size + paged_kv_last_page_len[batch_idx];
  if (token_idx >= valid_len) {
    return;
  }

  const int32_t local_page = token_idx / page_size;
  const int32_t entry_idx = token_idx - local_page * page_size;
  const int32_t page_iter = paged_kv_indptr[batch_idx] + local_page;
  const int32_t page_idx = paged_kv_indices[page_iter];

  const int32_t packed_head_dim = head_dim / 2;
  const int64_t k_base =
      (((int64_t)page_idx * page_size + entry_idx) * num_heads + head_idx) * packed_head_dim;
  const int64_t q_base = ((int64_t)batch_idx * num_heads + head_idx) * head_dim;
  const int64_t scale_idx = ((int64_t)page_idx * page_size + entry_idx) * num_heads + head_idx;

  const float scale = __half2float(quant_scales[scale_idx]);
  const float zero = __half2float(quant_zeros[scale_idx]);

  float partial = 0.0f;
  for (int32_t dim = tid; dim < head_dim; dim += blockDim.x) {
    const uint8_t packed = k_cache[k_base + dim / 2];
    const uint8_t q4 = (dim % 2 == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
    const float k_val = static_cast<float>(q4) * scale + zero;
    partial += __half2float(q[q_base + dim]) * k_val;
  }

  __shared__ float smem[128];
  smem[tid] = partial;
  __syncthreads();

  for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      smem[tid] += smem[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    o[((int64_t)batch_idx * num_heads + head_idx) * kv_len + token_idx] = __float2half(smem[0]);
  }
}

template <typename DTypeQ>
__global__ void SparseBatchedInt4GEMVKernel(DTypeQ* q, DTypeQ* o, uint8_t* k_cache,
                                           DTypeQ* quant_scales, DTypeQ* quant_zeros,
                                           int32_t* selected_indices, uint8_t* selected_valid,
                                           int32_t batch_size, int32_t num_heads,
                                           int32_t head_dim, int32_t seq_len,
                                           int32_t max_selected) {
  constexpr int32_t tokens_per_block = 4;
  constexpr int32_t threads_per_token = 32;

  const int32_t lane = threadIdx.x % threads_per_token;
  const int32_t token_lane = threadIdx.x / threads_per_token;
  const int32_t selected_pos = blockIdx.x * tokens_per_block + token_lane;
  const int32_t head_idx = blockIdx.y;
  const int32_t batch_idx = blockIdx.z;

  if (batch_idx >= batch_size || head_idx >= num_heads || selected_pos >= max_selected) {
    return;
  }

  const int64_t selected_offset =
      ((int64_t)batch_idx * num_heads + head_idx) * max_selected + selected_pos;
  if (selected_valid[selected_offset] == 0) {
    if (lane == 0) {
      o[selected_offset] = __float2half(0.0f);
    }
    return;
  }

  const int32_t token_idx = selected_indices[selected_offset];
  const int32_t packed_head_dim = head_dim / 2;
  const int64_t k_base =
      (((int64_t)batch_idx * seq_len + token_idx) * num_heads + head_idx) * packed_head_dim;
  const int64_t q_base = ((int64_t)batch_idx * num_heads + head_idx) * head_dim;
  const int64_t scale_idx = ((int64_t)batch_idx * seq_len + token_idx) * num_heads + head_idx;

  const float scale = __half2float(quant_scales[scale_idx]);
  const float zero = __half2float(quant_zeros[scale_idx]);

  float partial = 0.0f;
  for (int32_t dim = lane; dim < head_dim; dim += threads_per_token) {
    const uint8_t packed = k_cache[k_base + dim / 2];
    const uint8_t q4 = (dim % 2 == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
    const float k_val = static_cast<float>(q4) * scale + zero;
    partial += __half2float(q[q_base + dim]) * k_val;
  }

  __shared__ float smem[128];
  smem[threadIdx.x] = partial;
  __syncthreads();

  const int32_t smem_base = token_lane * threads_per_token;
  for (int32_t stride = threads_per_token / 2; stride > 0; stride >>= 1) {
    if (lane < stride) {
      smem[smem_base + lane] += smem[smem_base + lane + stride];
    }
    __syncthreads();
  }

  if (lane == 0) {
    o[selected_offset] = __float2half(smem[smem_base]);
  }
}

void single_sparse_gemv(at::Tensor q, at::Tensor o, at::Tensor k) {
  // q: [num_qo_heads, head_dim]
  // o: [num_qo_heads, seq_len]
  // k: [seq_len, num_kv_heads, head_dim] (NHD)

  int64_t seq_len = k.size(0);
  int64_t num_qo_heads = q.size(0);
  int64_t num_kv_heads = k.size(1);
  int64_t head_dim = q.size(1);

  QKVLayout kv_layout = static_cast<QKVLayout>(0);  // NHD

  bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE(q.scalar_type(), c_type, [&] {
    cudaError_t status = SingleSparseGEMV<128, c_type, c_type, c_type, int32_t>(
        static_cast<c_type*>(q.data_ptr()), static_cast<c_type*>(k.data_ptr()),
        static_cast<c_type*>(o.data_ptr()), num_qo_heads, num_kv_heads, seq_len);
    TORCH_CHECK(status == cudaSuccess, "SingleSparseGEMV failed with error code ",
                cudaGetErrorString(status));
    return true;
  });
}

void batched_sparse_gemv(at::Tensor q, at::Tensor o, at::Tensor k_cache,
                         at::Tensor paged_kv_indices, at::Tensor paged_kv_indptr,
                         at::Tensor paged_kv_last_page_len) {
  int64_t batch_size = q.size(0);
  int64_t num_heads = q.size(1);
  int64_t head_dim = q.size(2);
  int64_t page_size = k_cache.size(1);

  const int64_t* kv_cache_strides = nullptr;
  auto k_strides = k_cache.strides();
  kv_cache_strides = k_strides.data();

  QKVLayout kv_layout = static_cast<QKVLayout>(0);  // NHD

  bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE(q.scalar_type(), c_type, [&] {
    paged_kv_t<c_type, int32_t> paged_kv(num_heads, page_size, head_dim, batch_size, kv_layout,
                                         static_cast<c_type*>(k_cache.data_ptr()),
                                         nullptr,  // v cache is not used
                                         kv_cache_strides,
                                         static_cast<int32_t*>(paged_kv_indices.data_ptr()),
                                         static_cast<int32_t*>(paged_kv_indptr.data_ptr()),
                                         static_cast<int32_t*>(paged_kv_last_page_len.data_ptr()));

    cudaError_t status = BatchedSparseGEMV<128, c_type, c_type, c_type, int32_t>(
        static_cast<c_type*>(q.data_ptr()), paged_kv, nullptr, nullptr,
        static_cast<c_type*>(o.data_ptr()), num_heads);
    TORCH_CHECK(status == cudaSuccess, "BatchedSparseGEMV failed with error code ",
                cudaGetErrorString(status));
    return true;
  });
}

void batched_sparse_gemv_int8_k(at::Tensor q, at::Tensor o, at::Tensor k_cache,
                                at::Tensor quant_scales, at::Tensor quant_zeros,
                                at::Tensor paged_kv_indices, at::Tensor paged_kv_indptr,
                                at::Tensor paged_kv_last_page_len) {
  int64_t batch_size = q.size(0);
  int64_t num_heads = q.size(1);
  int64_t head_dim = q.size(2);
  int64_t page_size = k_cache.size(1);

  const int64_t* kv_cache_strides = nullptr;
  auto k_strides = k_cache.strides();
  kv_cache_strides = k_strides.data();

  QKVLayout kv_layout = static_cast<QKVLayout>(0);  // NHD

  bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE(q.scalar_type(), c_type, [&] {
    // int8 k cache
    paged_kv_t<uint8_t, int32_t> paged_kv(num_heads, page_size, head_dim, batch_size, kv_layout,
                                          static_cast<uint8_t*>(k_cache.data_ptr()),
                                          nullptr,  // v cache is not used
                                          kv_cache_strides,
                                          static_cast<int32_t*>(paged_kv_indices.data_ptr()),
                                          static_cast<int32_t*>(paged_kv_indptr.data_ptr()),
                                          static_cast<int32_t*>(paged_kv_last_page_len.data_ptr()));

    cudaError_t status = BatchedSparseGEMV<128, c_type, uint8_t, c_type, int32_t>(
        static_cast<c_type*>(q.data_ptr()), paged_kv, static_cast<c_type*>(quant_scales.data_ptr()),
        static_cast<c_type*>(quant_zeros.data_ptr()), static_cast<c_type*>(o.data_ptr()),
        num_heads);
    TORCH_CHECK(status == cudaSuccess, "Estimate_attn_score failed with error code ",
                cudaGetErrorString(status));
    return true;
  });
}

void batched_sparse_gemv_int4_k(at::Tensor q, at::Tensor o, at::Tensor k_cache,
                                at::Tensor quant_scales, at::Tensor quant_zeros,
                                at::Tensor paged_kv_indices, at::Tensor paged_kv_indptr,
                                at::Tensor paged_kv_last_page_len) {
  int64_t batch_size = q.size(0);
  int64_t num_heads = q.size(1);
  int64_t head_dim = q.size(2);
  int64_t page_size = k_cache.size(1);
  int64_t kv_len = o.size(2);

  bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE(q.scalar_type(), c_type, [&] {
    dim3 nblks(kv_len, num_heads, batch_size);
    dim3 nthrs(128);
    SimpleBatchedInt4GEMVKernel<c_type><<<nblks, nthrs, 0, at::cuda::getCurrentCUDAStream()>>>(
        static_cast<c_type*>(q.data_ptr()), static_cast<c_type*>(o.data_ptr()),
        static_cast<uint8_t*>(k_cache.data_ptr()), static_cast<c_type*>(quant_scales.data_ptr()),
        static_cast<c_type*>(quant_zeros.data_ptr()),
        static_cast<int32_t*>(paged_kv_indices.data_ptr()),
        static_cast<int32_t*>(paged_kv_indptr.data_ptr()),
        static_cast<int32_t*>(paged_kv_last_page_len.data_ptr()), batch_size, num_heads, head_dim,
        page_size, kv_len);
    cudaError_t status = cudaGetLastError();
    TORCH_CHECK(status == cudaSuccess, "SimpleBatchedInt4GEMVKernel failed with error code ",
                cudaGetErrorString(status));
    return true;
  });
}

void sparse_batched_gemv_int4_k(at::Tensor q, at::Tensor o, at::Tensor k_cache,
                                at::Tensor quant_scales, at::Tensor quant_zeros,
                                at::Tensor selected_indices, at::Tensor selected_valid) {
  int64_t batch_size = q.size(0);
  int64_t num_heads = q.size(1);
  int64_t head_dim = q.size(2);
  int64_t seq_len = k_cache.size(0) / batch_size;
  int64_t max_selected = selected_indices.size(2);

  bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE(q.scalar_type(), c_type, [&] {
    constexpr int32_t tokens_per_block = 4;
    dim3 nblks((max_selected + tokens_per_block - 1) / tokens_per_block, num_heads, batch_size);
    dim3 nthrs(128);
    SparseBatchedInt4GEMVKernel<c_type><<<nblks, nthrs, 0, at::cuda::getCurrentCUDAStream()>>>(
        static_cast<c_type*>(q.data_ptr()), static_cast<c_type*>(o.data_ptr()),
        static_cast<uint8_t*>(k_cache.data_ptr()), static_cast<c_type*>(quant_scales.data_ptr()),
        static_cast<c_type*>(quant_zeros.data_ptr()),
        static_cast<int32_t*>(selected_indices.data_ptr()),
        static_cast<uint8_t*>(selected_valid.data_ptr()), batch_size, num_heads, head_dim, seq_len,
        max_selected);
    cudaError_t status = cudaGetLastError();
    TORCH_CHECK(status == cudaSuccess, "SparseBatchedInt4GEMVKernel failed with error code ",
                cudaGetErrorString(status));
    return true;
  });
}

void quest_sparse_gemv(at::Tensor q, at::Tensor o, at::Tensor k_cache, at::Tensor v_cache,
                       at::Tensor paged_kv_indices, at::Tensor paged_kv_indptr,
                       at::Tensor paged_kv_last_page_len) {
  int64_t batch_size = q.size(0);
  int64_t num_heads = q.size(1);
  int64_t head_dim = q.size(2);
  int64_t page_size = k_cache.size(1);

  const int64_t* kv_cache_strides = nullptr;
  auto k_strides = k_cache.strides();
  kv_cache_strides = k_strides.data();

  QKVLayout kv_layout = static_cast<QKVLayout>(0);  // NHD

  // size_t output_len = (metadata_indices.size(0) - 1) * page_size + metadata_last_page_len - 1;
  // at::Tensor o = torch::empty(
  // {static_cast<signed long>(num_heads), static_cast<signed long>(output_len)}, q.options());

  bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE(q.scalar_type(), c_type, [&] {
    paged_kv_t<c_type, int32_t> paged_kv(num_heads, page_size, head_dim, batch_size, kv_layout,
                                         static_cast<c_type*>(k_cache.data_ptr()),
                                         static_cast<c_type*>(v_cache.data_ptr()), kv_cache_strides,
                                         static_cast<int32_t*>(paged_kv_indices.data_ptr()),
                                         static_cast<int32_t*>(paged_kv_indptr.data_ptr()),
                                         static_cast<int32_t*>(paged_kv_last_page_len.data_ptr()));

    cudaError_t status = MaxPossibleSampleWithPagedKVCache<128, c_type, c_type, c_type, int32_t>(
        static_cast<c_type*>(q.data_ptr()), paged_kv, static_cast<c_type*>(o.data_ptr()),
        num_heads);
    TORCH_CHECK(status == cudaSuccess, "Estimate_attn_score failed with error code ",
                cudaGetErrorString(status));
    return true;
  });
  TORCH_CHECK(success, "Estimate_attn_score failed to dispatch with dtype ", q.scalar_type());
}

/*!
 * \brief Select Top-k value in a batched tensor
 * \tparam T The data type
 * \tparam idxT The index type
 * \tparam num_heads batch size
 * \param in [batch_size, len] data of tensor
 * \param in_idx [batch_size, len] index of tensor
 * \param len column width
 * \param k number of top-k elements to select
 * \param out [batch_size, k] output data
 * \param out_idx [batch_size, k] output index
 * \param greater whether to select top-k or bottom-k
 */
void raft_topk(at::Tensor in, at::Tensor in_idx, at::Tensor out, at::Tensor out_idx, at::Tensor buf,
               uint32_t k) {
  auto batch_size = in.size(0);
  auto seq_len = in.size(1);

  bool success = DISPATCH_PYTORCH_DTYPE_TO_CTYPE(in.scalar_type(), c_type, [&] {
    raft::decode_select_k<c_type, int32_t>(
        static_cast<c_type*>(in.data_ptr()), static_cast<int32_t*>(in_idx.data_ptr()),
        static_cast<char*>(buf.data_ptr()), batch_size, seq_len, k,
        static_cast<c_type*>(out.data_ptr()), static_cast<int32_t*>(out_idx.data_ptr()), true);
    return true;
  });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("single_sparse_gemv", &single_sparse_gemv, "Single Sparse GEMV");
  m.def("batched_sparse_gemv", &batched_sparse_gemv, "Batched Sparse (w/ Paged Key Cache) GEMV");
  m.def("batched_sparse_gemv_int8_k", &batched_sparse_gemv_int8_k,
        "Batched Sparse (w/ Int8 Quantized Paged Key Cache) GEMV");
  m.def("batched_sparse_gemv_int4_k", &batched_sparse_gemv_int4_k,
        "Batched Sparse (w/ Int4 Quantized Paged Key Cache) GEMV");
  m.def("sparse_batched_gemv_int4_k", &sparse_batched_gemv_int4_k,
        "Batched Sparse Int4 GEMV over selected token indices");
  m.def("quest_sparse_gemv", &quest_sparse_gemv, "Quest Sparse GEMV");
  m.def("raft_topk", &raft_topk, "TopK");
}
