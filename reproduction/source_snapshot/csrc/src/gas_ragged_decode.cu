#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

template<typename T> __device__ __forceinline__ float as_float(T x);
template<> __device__ __forceinline__ float as_float<half>(half x) { return __half2float(x); }
template<> __device__ __forceinline__ float as_float<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }
template<typename T> __device__ __forceinline__ T as_output(float x);
template<> __device__ __forceinline__ half as_output<half>(float x) { return __float2half(x); }
template<> __device__ __forceinline__ __nv_bfloat16 as_output<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

// jobs, counts and offsets are a validated fixed plan. A short head writes
// its output directly; only a multi-part head writes FP32 partial states.
template<typename T, int Segments>
__global__ void ragged_partial(const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v,
                               const int32_t* __restrict__ shared, const int32_t* __restrict__ exceptions,
                               const int32_t* __restrict__ sc, const int32_t* __restrict__ ec,
                               const int32_t* __restrict__ jobs, const int32_t* __restrict__ parts,
                               const int32_t* __restrict__ offsets, float* __restrict__ partial,
                               T* __restrict__ out, int heads, int kv_heads, int length, int sw, int ew) {
  const int lane = threadIdx.x & 15, seg = threadIdx.x >> 4;
  const int row = jobs[blockIdx.x * 2], part = jobs[blockIdx.x * 2 + 1];
  const int batch = row / heads, head = row % heads, kvhead = head / 4;
  const int nparts = parts[row];
  const int64_t qb = static_cast<int64_t>(row) * 128;
  const int64_t kb = (static_cast<int64_t>(batch) * kv_heads + kvhead) * length * 128;
  const int64_t sb = (static_cast<int64_t>(batch) * kv_heads + kvhead) * sw;
  const int64_t eb = static_cast<int64_t>(row) * ew;
  const int ns = sc[static_cast<int64_t>(batch) * kv_heads + kvhead], ne = ec[row];
  const uint4 qr = *reinterpret_cast<const uint4*>(q + qb + lane * 8);
  const T* qh = reinterpret_cast<const T*>(&qr);
  float query[8], acc[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) { query[i] = as_float<T>(qh[i]); acc[i] = 0; }
  float m = -INFINITY, sum = 0;
  for (int which = 0; which < 2; ++which) {
    const int count = which == 0 ? ns : ne;
    const int32_t* index = which == 0 ? shared + sb : exceptions + eb;
    const int span = (count + nparts - 1) / nparts;
    const int start = part * span, end = min(count, start + span);
    for (int pos = start + seg; pos < end; pos += Segments) {
      const int token = index[pos];
      if (token < 0 || token >= length) continue;
      const int64_t base = kb + static_cast<int64_t>(token) * 128 + lane * 8;
      const uint4 kr = *reinterpret_cast<const uint4*>(k + base);
      const T* kh = reinterpret_cast<const T*>(&kr);
      float dot = 0;
      #pragma unroll
      for (int i = 0; i < 8; ++i) dot += query[i] * as_float<T>(kh[i]);
      const unsigned active = __activemask();
      for (int d = 8; d; d >>= 1) dot += __shfl_down_sync(active, dot, d, 16);
      dot = __shfl_sync(active, dot, 0, 16) * 0.08838834764831845f;
      const float next = fmaxf(m, dot), a = __expf(m - next), b = __expf(dot - next);
      const uint4 vr = *reinterpret_cast<const uint4*>(v + base);
      const T* vh = reinterpret_cast<const T*>(&vr);
      #pragma unroll
      for (int i = 0; i < 8; ++i) acc[i] = acc[i] * a + b * as_float<T>(vh[i]);
      sum = sum * a + b;
      m = next;
    }
  }
  __shared__ float states[Segments][8][16], meta[Segments][2];
  #pragma unroll
  for (int i = 0; i < 8; ++i) states[seg][i][lane] = acc[i];
  if (!lane) { meta[seg][0] = m; meta[seg][1] = sum; }
  __syncthreads();
  if (seg == 0) {
    float maximum = -INFINITY;
    #pragma unroll
    for (int s = 0; s < Segments; ++s) maximum = fmaxf(maximum, meta[s][0]);
    float denom = 0, result[8] = {0};
    #pragma unroll
    for (int s = 0; s < Segments; ++s) {
      if (meta[s][1] == 0) continue;
      const float weight = __expf(meta[s][0] - maximum);
      denom += meta[s][1] * weight;
      #pragma unroll
      for (int i = 0; i < 8; ++i) result[i] += states[s][i][lane] * weight;
    }
    if (nparts == 1) {
      uint4 raw;
      T* value = reinterpret_cast<T*>(&raw);
      #pragma unroll
      for (int i = 0; i < 8; ++i) value[i] = as_output<T>(result[i] / denom);
      *reinterpret_cast<uint4*>(out + qb + lane * 8) = raw;
    } else {
      const int64_t base = (static_cast<int64_t>(offsets[row]) + part) * 130;
      #pragma unroll
      for (int i = 0; i < 8; ++i) partial[base + lane * 8 + i] = result[i];
      if (!lane) { partial[base + 128] = maximum; partial[base + 129] = denom; }
    }
  }
}

template<typename T>
__global__ void ragged_merge(const float* __restrict__ partial, T* __restrict__ out,
                             const int32_t* __restrict__ long_rows, const int32_t* __restrict__ parts,
                             const int32_t* __restrict__ offsets) {
  const int lane = threadIdx.x, row = long_rows[blockIdx.x], nparts = parts[row];
  const int64_t base = static_cast<int64_t>(offsets[row]) * 130;
  float maximum = -INFINITY;
  for (int p = 0; p < nparts; ++p) maximum = fmaxf(maximum, partial[base + p * 130 + 128]);
  float denom = 0, result[8] = {0};
  for (int p = 0; p < nparts; ++p) {
    const int64_t pb = base + p * 130;
    const float part_sum = partial[pb + 129];
    if (part_sum == 0) continue;
    const float weight = __expf(partial[pb + 128] - maximum);
    denom += part_sum * weight;
    #pragma unroll
    for (int i = 0; i < 8; ++i) result[i] += partial[pb + lane * 8 + i] * weight;
  }
  uint4 raw;
  T* value = reinterpret_cast<T*>(&raw);
  #pragma unroll
  for (int i = 0; i < 8; ++i) value[i] = as_output<T>(result[i] / denom);
  *reinterpret_cast<uint4*>(out + static_cast<int64_t>(row) * 128 + lane * 8) = raw;
}

template<typename T>
void run(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor shared, torch::Tensor ex,
         torch::Tensor sc, torch::Tensor ec, torch::Tensor jobs, torch::Tensor parts,
         torch::Tensor offsets, torch::Tensor long_rows, torch::Tensor out, int segments, int64_t partial_count) {
  auto partial = torch::empty({partial_count > 0 ? partial_count : 1, 130}, q.options().dtype(at::kFloat));
  const auto qp = reinterpret_cast<const T*>(q.data_ptr()), kp = reinterpret_cast<const T*>(k.data_ptr()), vp = reinterpret_cast<const T*>(v.data_ptr());
  const auto op = reinterpret_cast<T*>(out.data_ptr());
  const auto stream = at::cuda::getCurrentCUDAStream();
  #define RUN(N) ragged_partial<T, N><<<jobs.size(0), N * 16, 0, stream>>>(qp, kp, vp, shared.data_ptr<int32_t>(), ex.data_ptr<int32_t>(), sc.data_ptr<int32_t>(), ec.data_ptr<int32_t>(), jobs.data_ptr<int32_t>(), parts.data_ptr<int32_t>(), offsets.data_ptr<int32_t>(), partial.data_ptr<float>(), op, q.size(1), k.size(1), k.size(2), shared.size(2), ex.size(2))
  if (segments == 4) { RUN(4); } else if (segments == 8) { RUN(8); } else { RUN(16); }
  #undef RUN
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (long_rows.numel() > 0) {
    ragged_merge<T><<<long_rows.numel(), 16, 0, stream>>>(partial.data_ptr<float>(), op, long_rows.data_ptr<int32_t>(), parts.data_ptr<int32_t>(), offsets.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

torch::Tensor decode(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor shared, torch::Tensor ex,
                     torch::Tensor sc, torch::Tensor ec, torch::Tensor jobs, torch::Tensor parts,
                     torch::Tensor offsets, torch::Tensor long_rows, int segments, int64_t partial_count) {
  for (const auto& x : {q, k, v, shared, ex, sc, ec, jobs, parts, offsets, long_rows}) {
    TORCH_CHECK(x.is_cuda() && x.device() == q.device() && x.is_contiguous(), "all tensors must be contiguous on one CUDA device");
  }
  TORCH_CHECK((q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16) && q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(), "FP16 or BF16 Q/K/V required");
  for (const auto& x : {shared, ex, sc, ec, jobs, parts, offsets, long_rows}) TORCH_CHECK(x.scalar_type() == at::kInt, "int32 metadata required");
  TORCH_CHECK(q.dim() == 3 && k.dim() == 4 && k.sizes() == v.sizes() && q.size(0) == k.size(0) && q.size(1) == 4 * k.size(1) && q.size(2) == 128 && k.size(3) == 128, "invalid Q/K/V shape");
  TORCH_CHECK(shared.dim() == 3 && shared.size(0) == q.size(0) && shared.size(1) == k.size(1) && ex.dim() == 3 && ex.size(0) == q.size(0) && ex.size(1) == q.size(1), "invalid support shape");
  TORCH_CHECK(sc.dim() == 2 && sc.size(0) == q.size(0) && sc.size(1) == k.size(1) && ec.dim() == 2 && ec.sizes() == q.sizes().slice(0, 2), "invalid count shape");
  TORCH_CHECK(jobs.dim() == 2 && jobs.size(1) == 2 && jobs.size(0) >= q.size(0) * q.size(1) && parts.dim() == 2 && parts.sizes() == ec.sizes() && offsets.sizes() == ec.sizes() && long_rows.dim() == 1 && long_rows.numel() <= ec.numel(), "invalid plan shape");
  TORCH_CHECK(k.size(2) > 0 && shared.size(2) + ex.size(2) > 0 && partial_count >= 0, "empty support or invalid scratch size");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(k.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(v.data_ptr()) % 16 == 0, "16-byte alignment required");
  TORCH_CHECK(segments == 4 || segments == 8 || segments == 16, "unsupported segment count");
  const c10::cuda::CUDAGuard guard(q.device());
  auto out = torch::empty_like(q);
  if (q.scalar_type() == at::kHalf) run<half>(q, k, v, shared, ex, sc, ec, jobs, parts, offsets, long_rows, out, segments, partial_count);
  else run<__nv_bfloat16>(q, k, v, shared, ex, sc, ec, jobs, parts, offsets, long_rows, out, segments, partial_count);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("decode", &decode); }
