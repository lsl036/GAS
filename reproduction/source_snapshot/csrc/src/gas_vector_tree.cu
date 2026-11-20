#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// Each 16-thread subgroup handles a strided token subset; an in-CTA tree
// merges their online softmax states without a global intermediate buffer.
template<int Segments>
__global__ void gas_vector_tree(const half* __restrict__ q, const half* __restrict__ k,
                                const half* __restrict__ v, const int32_t* __restrict__ shared,
                                const int32_t* __restrict__ exceptions, half* __restrict__ out,
                                int heads, int kv_heads, int length, int sw, int ew) {
  const int lane = threadIdx.x & 15, segment = threadIdx.x >> 4;
  const int batch = blockIdx.x, head = blockIdx.y, kvhead = head / 4;
  const int64_t qb = (static_cast<int64_t>(batch) * heads + head) * 128;
  const int64_t kb = (static_cast<int64_t>(batch) * kv_heads + kvhead) * length * 128;
  const int64_t sb = (static_cast<int64_t>(batch) * kv_heads + kvhead) * sw;
  const int64_t eb = (static_cast<int64_t>(batch) * heads + head) * ew;
  const uint4 qr = *reinterpret_cast<const uint4*>(q + qb + lane * 8);
  const half* qh = reinterpret_cast<const half*>(&qr);
  float query[8], acc[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) { query[i] = __half2float(qh[i]); acc[i] = 0; }
  float maximum = -INFINITY, sum = 0;
  for (int pos = segment; pos < sw + ew; pos += Segments) {
    const int token = pos < sw ? shared[sb + pos] : exceptions[eb + pos - sw];
    if (token < 0 || token >= length) continue;
    const int64_t base = kb + static_cast<int64_t>(token) * 128 + lane * 8;
    const uint4 kr = *reinterpret_cast<const uint4*>(k + base);
    const half* kh = reinterpret_cast<const half*>(&kr);
    float dot = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) dot += query[i] * __half2float(kh[i]);
    const unsigned active = __activemask();
    for (int d = 8; d; d >>= 1) dot += __shfl_down_sync(active, dot, d, 16);
    dot = __shfl_sync(active, dot, 0, 16) * 0.08838834764831845f;
    const float next = fmaxf(maximum, dot), a = __expf(maximum - next), b = __expf(dot - next);
    const uint4 vr = *reinterpret_cast<const uint4*>(v + base);
    const half* vh = reinterpret_cast<const half*>(&vr);
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = acc[i] * a + b * __half2float(vh[i]);
    sum = sum * a + b;
    maximum = next;
  }
  __shared__ float states[Segments][8][16];
  __shared__ float meta[Segments][2];
  #pragma unroll
  for (int i = 0; i < 8; ++i) states[segment][i][lane] = acc[i];
  if (!lane) { meta[segment][0] = maximum; meta[segment][1] = sum; }
  __syncthreads();
  for (int offset = Segments / 2; offset; offset >>= 1) {
    if (segment < offset) {
      const float other_max = meta[segment + offset][0];
      const float other_sum = meta[segment + offset][1];
      const float next = fmaxf(maximum, other_max);
      const float a = sum > 0 ? __expf(maximum - next) : 0;
      const float b = other_sum > 0 ? __expf(other_max - next) : 0;
      #pragma unroll
      for (int i = 0; i < 8; ++i) {
        acc[i] = acc[i] * a + states[segment + offset][i][lane] * b;
        states[segment][i][lane] = acc[i];
      }
      sum = sum * a + other_sum * b;
      maximum = next;
      if (!lane) { meta[segment][0] = maximum; meta[segment][1] = sum; }
    }
    __syncthreads();
  }
  if (segment == 0) {
    uint4 raw;
    half* result = reinterpret_cast<half*>(&raw);
    #pragma unroll
    for (int i = 0; i < 8; ++i) result[i] = __float2half(acc[i] / sum);
    *reinterpret_cast<uint4*>(out + qb + lane * 8) = raw;
  }
}

torch::Tensor decode(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                     torch::Tensor shared, torch::Tensor exceptions, int segments) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && shared.is_cuda() && exceptions.is_cuda(), "CUDA required");
  TORCH_CHECK(q.device() == k.device() && q.device() == v.device() && q.device() == shared.device() && q.device() == exceptions.device(), "device mismatch");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && shared.is_contiguous() && exceptions.is_contiguous(), "contiguous required");
  TORCH_CHECK(q.scalar_type() == at::kHalf && k.scalar_type() == at::kHalf && v.scalar_type() == at::kHalf && shared.scalar_type() == at::kInt && exceptions.scalar_type() == at::kInt, "invalid dtype");
  TORCH_CHECK(q.dim() == 3 && k.dim() == 4 && k.sizes() == v.sizes() && q.size(0) == k.size(0) && q.size(1) == 4 * k.size(1) && q.size(2) == 128 && k.size(3) == 128, "invalid Q/K/V shape");
  TORCH_CHECK(shared.dim() == 3 && shared.size(0) == q.size(0) && shared.size(1) == k.size(1) && exceptions.dim() == 3 && exceptions.size(0) == q.size(0) && exceptions.size(1) == q.size(1), "invalid support shape");
  TORCH_CHECK(k.size(2) > 0 && shared.size(2) + exceptions.size(2) > 0, "empty support");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(k.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(v.data_ptr()) % 16 == 0, "16-byte alignment required");
  TORCH_CHECK(segments == 8 || segments == 16 || segments == 32, "unsupported segment count");
  const c10::cuda::CUDAGuard guard(q.device());
  auto out = torch::empty_like(q);
  const auto qp = reinterpret_cast<const half*>(q.data_ptr<at::Half>());
  const auto kp = reinterpret_cast<const half*>(k.data_ptr<at::Half>());
  const auto vp = reinterpret_cast<const half*>(v.data_ptr<at::Half>());
  const auto op = reinterpret_cast<half*>(out.data_ptr<at::Half>());
  const dim3 grid(q.size(0), q.size(1));
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (segments == 8)
    gas_vector_tree<8><<<grid, 128, 0, stream>>>(qp, kp, vp, shared.data_ptr<int32_t>(), exceptions.data_ptr<int32_t>(), op, q.size(1), k.size(1), k.size(2), shared.size(2), exceptions.size(2));
  else if (segments == 16)
    gas_vector_tree<16><<<grid, 256, 0, stream>>>(qp, kp, vp, shared.data_ptr<int32_t>(), exceptions.data_ptr<int32_t>(), op, q.size(1), k.size(1), k.size(2), shared.size(2), exceptions.size(2));
  else
    gas_vector_tree<32><<<grid, 512, 0, stream>>>(qp, kp, vp, shared.data_ptr<int32_t>(), exceptions.data_ptr<int32_t>(), op, q.size(1), k.size(1), k.size(2), shared.size(2), exceptions.size(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("decode", &decode); }
