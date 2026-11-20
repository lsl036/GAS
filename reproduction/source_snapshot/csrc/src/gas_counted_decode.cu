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
template<typename T> __device__ __forceinline__ T from_float(float x);
template<> __device__ __forceinline__ half from_float<half>(float x) { return __float2half(x); }
template<> __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

// Each row must be a valid nonnegative prefix followed only by negative pads.
__global__ void count_padded_prefixes(const int32_t* __restrict__ shared, const int32_t* __restrict__ exceptions,
                                      int32_t* __restrict__ sc, int32_t* __restrict__ ec,
                                      int shared_rows, int exception_rows, int sw, int ew) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= shared_rows + exception_rows) return;
  const bool group = row < shared_rows;
  const int local = group ? row : row - shared_rows;
  const int width = group ? sw : ew;
  const int32_t* index = group ? shared + static_cast<int64_t>(local) * sw : exceptions + static_cast<int64_t>(local) * ew;
  int lo = 0, hi = width;
  while (lo < hi) {
    const int mid = lo + (hi - lo) / 2;
    if (index[mid] >= 0) lo = mid + 1;
    else hi = mid;
  }
  if (group) sc[local] = lo;
  else ec[local] = lo;
}

std::vector<torch::Tensor> count_prefix(torch::Tensor shared, torch::Tensor exceptions) {
  TORCH_CHECK(shared.is_cuda() && exceptions.is_cuda() && shared.device() == exceptions.device(), "matching CUDA devices required");
  TORCH_CHECK(shared.is_contiguous() && exceptions.is_contiguous() && shared.scalar_type() == at::kInt && exceptions.scalar_type() == at::kInt, "contiguous int32 indices required");
  TORCH_CHECK(shared.dim() == 3 && exceptions.dim() == 3 && shared.size(0) == exceptions.size(0) && exceptions.size(1) == 4 * shared.size(1), "invalid support shape");
  const c10::cuda::CUDAGuard guard(shared.device());
  auto sc = torch::empty({shared.size(0), shared.size(1)}, shared.options());
  auto ec = torch::empty({exceptions.size(0), exceptions.size(1)}, exceptions.options());
  const int gs = sc.numel(), hs = ec.numel(), total = gs + hs;
  TORCH_CHECK(total > 0, "empty batch or heads");
  count_padded_prefixes<<<(total + 127) / 128, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      shared.data_ptr<int32_t>(), exceptions.data_ptr<int32_t>(), sc.data_ptr<int32_t>(), ec.data_ptr<int32_t>(),
      gs, hs, shared.size(2), exceptions.size(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {sc, ec};
}

// Per-row counts delimit packed valid prefixes. Each 16-lane subgroup handles
// one strided token stream, followed by an in-block online-softmax merge.
template<typename T>
__device__ __forceinline__ void copy_async_16(T* dst, const T* src) {
#if __CUDA_ARCH__ >= 800
  const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(address), "l"(src) : "memory");
#else
  *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
#endif
}

__device__ __forceinline__ void async_commit() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

__device__ __forceinline__ void async_wait() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
}

template<typename T, int Segments, bool InlineCounts, bool TreeMerge, bool Async = false>
__global__ void counted_decode(const T* __restrict__ q, const T* __restrict__ k,
                               const T* __restrict__ v, const int32_t* __restrict__ shared,
                               const int32_t* __restrict__ exceptions, const int32_t* __restrict__ shared_counts,
                               const int32_t* __restrict__ exception_counts, T* __restrict__ out,
                               int heads, int kv_heads, int length, int sw, int ew) {
  const int lane = threadIdx.x & 15, segment = threadIdx.x >> 4;
  const int batch = blockIdx.x, head = blockIdx.y, kvhead = head / 4;
  const int64_t qb = (static_cast<int64_t>(batch) * heads + head) * 128;
  const int64_t kb = (static_cast<int64_t>(batch) * kv_heads + kvhead) * length * 128;
  const int64_t sb = (static_cast<int64_t>(batch) * kv_heads + kvhead) * sw;
  const int64_t eb = (static_cast<int64_t>(batch) * heads + head) * ew;
  __shared__ int inline_counts[2];
  if (InlineCounts) {
    if (threadIdx.x < 2) {
      const bool group = threadIdx.x == 0;
      const int32_t* index = group ? shared + sb : exceptions + eb;
      int lo = 0, hi = group ? sw : ew;
      while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (index[mid] >= 0) lo = mid + 1;
        else hi = mid;
      }
      inline_counts[threadIdx.x] = lo;
    }
    __syncthreads();
  }
  const int ns = InlineCounts ? inline_counts[0] : shared_counts[static_cast<int64_t>(batch) * kv_heads + kvhead];
  const int ne = InlineCounts ? inline_counts[1] : exception_counts[static_cast<int64_t>(batch) * heads + head];
  const uint4 qr = *reinterpret_cast<const uint4*>(q + qb + lane * 8);
  const T* qh = reinterpret_cast<const T*>(&qr);
  float query[8], acc[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) { query[i] = as_float<T>(qh[i]); acc[i] = 0; }
  float m = -INFINITY, sum = 0;
  if constexpr (Async) {
    __shared__ __align__(16) T k_tiles[2][Segments][128];
    __shared__ __align__(16) T v_tiles[2][Segments][128];
    const int total = ns + ne;
    const int rounds = (total + Segments - 1) / Segments;
    auto prefetch = [&](int round, int buffer) {
      const int pos = round * Segments + segment;
      if (pos < total) {
        const int token = pos < ns ? shared[sb + pos] : exceptions[eb + pos - ns];
        if (token >= 0 && token < length) {
          const int64_t base = kb + static_cast<int64_t>(token) * 128 + lane * 8;
          copy_async_16(&k_tiles[buffer][segment][lane * 8], k + base);
          copy_async_16(&v_tiles[buffer][segment][lane * 8], v + base);
        }
      }
      async_commit();
    };
    if (rounds) prefetch(0, 0);
    for (int round = 0; round < rounds; ++round) {
      async_wait();
      __syncthreads();
      if (round + 1 < rounds) prefetch(round + 1, (round + 1) & 1);
      const int pos = round * Segments + segment;
      if (pos < total) {
        const int token = pos < ns ? shared[sb + pos] : exceptions[eb + pos - ns];
        if (token >= 0 && token < length) {
          const T* kh = &k_tiles[round & 1][segment][lane * 8];
          const T* vh = &v_tiles[round & 1][segment][lane * 8];
          float dot = 0;
          #pragma unroll
          for (int i = 0; i < 8; ++i) dot += query[i] * as_float<T>(kh[i]);
          const unsigned active = __activemask();
          for (int d = 8; d; d >>= 1) dot += __shfl_down_sync(active, dot, d, 16);
          dot = __shfl_sync(active, dot, 0, 16) * 0.08838834764831845f;
          const float next = fmaxf(m, dot), a = __expf(m - next), b = __expf(dot - next);
          #pragma unroll
          for (int i = 0; i < 8; ++i) acc[i] = acc[i] * a + b * as_float<T>(vh[i]);
          sum = sum * a + b;
          m = next;
        }
      }
      __syncthreads();
    }
  } else for (int which = 0; which < 2; ++which) {
    const int count = which == 0 ? ns : ne;
    const int32_t* index = which == 0 ? shared + sb : exceptions + eb;
    for (int pos = segment; pos < count; pos += Segments) {
      const int token = index[pos];
      // Correct callers provide valid, distinct indices in each counted prefix.
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
  for (int i = 0; i < 8; ++i) states[segment][i][lane] = acc[i];
  if (!lane) { meta[segment][0] = m; meta[segment][1] = sum; }
  __syncthreads();
  if (TreeMerge) {
    for (int offset = Segments / 2; offset; offset >>= 1) {
      if (segment < offset) {
        const float other_m = meta[segment + offset][0];
        const float other_sum = meta[segment + offset][1];
        const float next = fmaxf(m, other_m);
        const float a = sum > 0 ? __expf(m - next) : 0;
        const float b = other_sum > 0 ? __expf(other_m - next) : 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
          acc[i] = acc[i] * a + states[segment + offset][i][lane] * b;
          states[segment][i][lane] = acc[i];
        }
        sum = sum * a + other_sum * b;
        m = next;
        if (!lane) { meta[segment][0] = m; meta[segment][1] = sum; }
      }
      __syncthreads();
    }
    if (segment == 0) {
      uint4 raw;
      T* output = reinterpret_cast<T*>(&raw);
      #pragma unroll
      for (int i = 0; i < 8; ++i) output[i] = from_float<T>(acc[i] / sum);
      *reinterpret_cast<uint4*>(out + qb + lane * 8) = raw;
    }
    return;
  }
  if (segment == 0) {
    float maximum = -INFINITY;
    #pragma unroll
    for (int s = 0; s < Segments; ++s) maximum = fmaxf(maximum, meta[s][0]);
    float denominator = 0, result[8] = {0};
    #pragma unroll
    for (int s = 0; s < Segments; ++s) {
      if (meta[s][1] == 0) continue;
      const float weight = __expf(meta[s][0] - maximum);
      denominator += meta[s][1] * weight;
      #pragma unroll
      for (int i = 0; i < 8; ++i) result[i] += states[s][i][lane] * weight;
    }
    uint4 raw;
    T* output = reinterpret_cast<T*>(&raw);
    #pragma unroll
    for (int i = 0; i < 8; ++i) output[i] = from_float<T>(result[i] / denominator);
    *reinterpret_cast<uint4*>(out + qb + lane * 8) = raw;
  }
}

template<typename T, bool InlineCounts, bool TreeMerge = false, bool Async = false>
void launch(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor shared,
            torch::Tensor exceptions, torch::Tensor sc, torch::Tensor ec, torch::Tensor out, int segments) {
  const auto qp = reinterpret_cast<const T*>(q.data_ptr());
  const auto kp = reinterpret_cast<const T*>(k.data_ptr());
  const auto vp = reinterpret_cast<const T*>(v.data_ptr());
  const auto op = reinterpret_cast<T*>(out.data_ptr());
  const dim3 grid(q.size(0), q.size(1));
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int32_t* scp = InlineCounts ? nullptr : sc.data_ptr<int32_t>();
  const int32_t* ecp = InlineCounts ? nullptr : ec.data_ptr<int32_t>();
  #define RUN(N) counted_decode<T, N, InlineCounts, TreeMerge, Async><<<grid, N * 16, 0, stream>>>(qp, kp, vp, shared.data_ptr<int32_t>(), exceptions.data_ptr<int32_t>(), scp, ecp, op, q.size(1), k.size(1), k.size(2), shared.size(2), exceptions.size(2))
  if (segments == 4) { RUN(4); } else if (segments == 8) { RUN(8); }
  else if (segments == 16) { RUN(16); } else { if constexpr (!Async) { RUN(32); } }
  #undef RUN
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor decode_impl(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor shared,
                          torch::Tensor exceptions, torch::Tensor sc, torch::Tensor ec, int segments, bool inline_counts, bool tree_merge = false, bool async = false) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && shared.is_cuda() && exceptions.is_cuda(), "CUDA required");
  TORCH_CHECK(q.device() == k.device() && q.device() == v.device() && q.device() == shared.device() && q.device() == exceptions.device(), "device mismatch");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && shared.is_contiguous() && exceptions.is_contiguous(), "contiguous required");
  TORCH_CHECK((q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16) && q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(), "FP16 or BF16 Q/K/V required");
  TORCH_CHECK(shared.scalar_type() == at::kInt && exceptions.scalar_type() == at::kInt, "int32 indices required");
  TORCH_CHECK(q.dim() == 3 && k.dim() == 4 && k.sizes() == v.sizes() && q.size(0) == k.size(0) && q.size(1) == 4 * k.size(1) && q.size(2) == 128 && k.size(3) == 128, "invalid Q/K/V shape");
  TORCH_CHECK(shared.dim() == 3 && shared.size(0) == q.size(0) && shared.size(1) == k.size(1) && exceptions.dim() == 3 && exceptions.size(0) == q.size(0) && exceptions.size(1) == q.size(1), "invalid index shape");
  if (!inline_counts) {
    TORCH_CHECK(sc.is_cuda() && ec.is_cuda() && q.device() == sc.device() && q.device() == ec.device() && sc.is_contiguous() && ec.is_contiguous() && sc.scalar_type() == at::kInt && ec.scalar_type() == at::kInt, "invalid count device, layout or dtype");
    TORCH_CHECK(sc.dim() == 2 && sc.size(0) == q.size(0) && sc.size(1) == k.size(1) && ec.dim() == 2 && ec.size(0) == q.size(0) && ec.size(1) == q.size(1), "invalid count shape");
  }
  TORCH_CHECK(k.size(2) > 0 && shared.size(2) + exceptions.size(2) > 0, "empty sequence or support");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(k.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(v.data_ptr()) % 16 == 0, "16-byte alignment required");
  TORCH_CHECK(segments == 4 || segments == 8 || segments == 16 || segments == 32, "unsupported segment count");
  TORCH_CHECK(!async || (inline_counts && !tree_merge && segments != 32), "async requires inline counts and 4, 8 or 16 segments");
  const c10::cuda::CUDAGuard guard(q.device());
  auto out = torch::empty_like(q);
  if (q.scalar_type() == at::kHalf) {
    if (async) launch<half, true, false, true>(q, k, v, shared, exceptions, sc, ec, out, segments);
    else if (tree_merge) launch<half, true, true>(q, k, v, shared, exceptions, sc, ec, out, segments);
    else if (inline_counts) launch<half, true>(q, k, v, shared, exceptions, sc, ec, out, segments);
    else launch<half, false>(q, k, v, shared, exceptions, sc, ec, out, segments);
  } else {
    if (async) launch<__nv_bfloat16, true, false, true>(q, k, v, shared, exceptions, sc, ec, out, segments);
    else if (tree_merge) launch<__nv_bfloat16, true, true>(q, k, v, shared, exceptions, sc, ec, out, segments);
    else if (inline_counts) launch<__nv_bfloat16, true>(q, k, v, shared, exceptions, sc, ec, out, segments);
    else launch<__nv_bfloat16, false>(q, k, v, shared, exceptions, sc, ec, out, segments);
  }
  return out;
}

torch::Tensor decode(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor shared,
                     torch::Tensor exceptions, torch::Tensor sc, torch::Tensor ec, int segments) {
  return decode_impl(q, k, v, shared, exceptions, sc, ec, segments, false);
}

torch::Tensor decode_prefix(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor shared,
                            torch::Tensor exceptions, int segments) {
  return decode_impl(q, k, v, shared, exceptions, torch::Tensor(), torch::Tensor(), segments, true);
}

torch::Tensor decode_prefix_tree(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor shared,
                                 torch::Tensor exceptions, int segments) {
  return decode_impl(q, k, v, shared, exceptions, torch::Tensor(), torch::Tensor(), segments, true, true);
}

torch::Tensor decode_prefix_async(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor shared,
                                  torch::Tensor exceptions, int segments) {
  return decode_impl(q, k, v, shared, exceptions, torch::Tensor(), torch::Tensor(), segments, true, false, true);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode", &decode);
  m.def("decode_prefix", &decode_prefix);
  m.def("decode_prefix_tree", &decode_prefix_tree);
  m.def("decode_prefix_async", &decode_prefix_async);
  m.def("count_prefix", &count_prefix);
}
