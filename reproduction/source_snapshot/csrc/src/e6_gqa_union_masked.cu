#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>

// One block per KV group. K/V tiles are loaded once and shared by the four
// query warps; each warp still applies its own membership mask.
__global__ void e6_union_masked(const half* __restrict__ q,
                                const half* __restrict__ k,
                                const half* __restrict__ v,
                                const int32_t* __restrict__ indices,
                                const bool* __restrict__ membership,
                                half* __restrict__ output,
                                int kv_heads, int length, int width) {
  constexpr int kDim = 128;
  constexpr int kGroup = 4;
  constexpr int kTile = 16;
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int batch = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int qhead = kv_head * kGroup + warp;
  const int heads = kv_heads * kGroup;
  const int64_t kv_base = (static_cast<int64_t>(batch) * kv_heads + kv_head) * length * kDim;
  const int64_t q_base = (static_cast<int64_t>(batch) * heads + qhead) * kDim;
  const int64_t index_base = (static_cast<int64_t>(batch) * kv_heads + kv_head) * width;
  const int64_t member_base = (static_cast<int64_t>(batch) * heads + qhead) * width;
  __shared__ half tile[kTile * kDim];
  extern __shared__ float scores[];
  float q0 = __half2float(q[q_base + lane]);
  float q1 = __half2float(q[q_base + lane + 32]);
  float q2 = __half2float(q[q_base + lane + 64]);
  float q3 = __half2float(q[q_base + lane + 96]);

  for (int start = 0; start < width; start += kTile) {
    for (int i = tid; i < kTile * kDim; i += blockDim.x) {
      int token_pos = start + i / kDim;
      int token = token_pos < width ? indices[index_base + token_pos] : -1;
      tile[i] = token >= 0 ? k[kv_base + static_cast<int64_t>(token) * kDim + i % kDim] : __float2half(0.0f);
    }
    __syncthreads();
    #pragma unroll
    for (int t = 0; t < kTile; ++t) {
      int token_pos = start + t;
      if (token_pos < width) {
        int base = t * kDim + lane;
        float dot = q0 * __half2float(tile[base]) + q1 * __half2float(tile[base + 32]) +
                    q2 * __half2float(tile[base + 64]) + q3 * __half2float(tile[base + 96]);
        for (int delta = 16; delta; delta >>= 1) dot += __shfl_down_sync(0xffffffff, dot, delta);
        if (lane == 0) {
          bool valid = indices[index_base + token_pos] >= 0 && membership[member_base + token_pos];
          scores[warp * width + token_pos] = valid ? dot * 0.08838834764831845f : -INFINITY;
        }
      }
    }
    __syncthreads();
  }

  float maximum = -INFINITY;
  for (int t = lane; t < width; t += 32) maximum = fmaxf(maximum, scores[warp * width + t]);
  for (int delta = 16; delta; delta >>= 1) maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, delta));
  maximum = __shfl_sync(0xffffffff, maximum, 0);
  float denominator = 0.0f;
  for (int t = lane; t < width; t += 32) {
    float weight = __expf(scores[warp * width + t] - maximum);
    scores[warp * width + t] = weight;
    denominator += weight;
  }
  for (int delta = 16; delta; delta >>= 1) denominator += __shfl_down_sync(0xffffffff, denominator, delta);
  denominator = __shfl_sync(0xffffffff, denominator, 0);
  __syncwarp();

  float o0 = 0, o1 = 0, o2 = 0, o3 = 0;
  for (int start = 0; start < width; start += kTile) {
    for (int i = tid; i < kTile * kDim; i += blockDim.x) {
      int token_pos = start + i / kDim;
      int token = token_pos < width ? indices[index_base + token_pos] : -1;
      tile[i] = token >= 0 ? v[kv_base + static_cast<int64_t>(token) * kDim + i % kDim] : __float2half(0.0f);
    }
    __syncthreads();
    #pragma unroll
    for (int t = 0; t < kTile; ++t) {
      int token_pos = start + t;
      if (token_pos < width) {
        float weight = scores[warp * width + token_pos] / denominator;
        int base = t * kDim + lane;
        o0 += weight * __half2float(tile[base]);
        o1 += weight * __half2float(tile[base + 32]);
        o2 += weight * __half2float(tile[base + 64]);
        o3 += weight * __half2float(tile[base + 96]);
      }
    }
    __syncthreads();
  }
  output[q_base + lane] = __float2half(o0);
  output[q_base + lane + 32] = __float2half(o1);
  output[q_base + lane + 64] = __float2half(o2);
  output[q_base + lane + 96] = __float2half(o3);
}

torch::Tensor e6_union_masked_decode(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                     torch::Tensor indices, torch::Tensor membership) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && indices.is_cuda() && membership.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(q.device() == k.device() && q.device() == v.device() && q.device() == indices.device() && q.device() == membership.device(), "all tensors must use one device");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && indices.is_contiguous() && membership.is_contiguous(), "contiguous tensors required");
  TORCH_CHECK(q.scalar_type() == at::kHalf && k.scalar_type() == at::kHalf && v.scalar_type() == at::kHalf, "FP16 required");
  TORCH_CHECK(indices.scalar_type() == at::kInt && membership.scalar_type() == at::kBool, "int32 indices and bool membership required");
  TORCH_CHECK(q.dim() == 3 && k.dim() == 4 && v.sizes() == k.sizes(), "invalid Q/K/V shape");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(1) == k.size(1) * 4 && q.size(2) == 128 && k.size(3) == 128, "expected GQA group=4, dim=128");
  TORCH_CHECK(indices.dim() == 3 && indices.size(0) == q.size(0) && indices.size(1) == k.size(1) && indices.size(2) > 0, "invalid indices");
  TORCH_CHECK(membership.dim() == 3 && membership.size(0) == q.size(0) && membership.size(1) == q.size(1) && membership.size(2) == indices.size(2), "invalid membership");
  size_t smem = static_cast<size_t>(4) * indices.size(2) * sizeof(float);
  TORCH_CHECK(smem <= 48 * 1024, "score storage exceeds shared memory limit");
  auto out = torch::empty_like(q);
  e6_union_masked<<<dim3(q.size(0), k.size(1)), 128, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()), reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(v.data_ptr<at::Half>()), indices.data_ptr<int32_t>(), membership.data_ptr<bool>(),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()), k.size(1), k.size(2), indices.size(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode", &e6_union_masked_decode, "GQA union decode with per-head membership");
}
