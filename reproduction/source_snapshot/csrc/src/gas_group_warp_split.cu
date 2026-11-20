#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

template<typename T> __device__ __forceinline__ float fp32(T x);
template<> __device__ __forceinline__ float fp32<half>(half x) { return __half2float(x); }
template<> __device__ __forceinline__ float fp32<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }
template<typename T> __device__ __forceinline__ T from_fp32(float x);
template<> __device__ __forceinline__ half from_fp32<half>(float x) { return __float2half(x); }
template<> __device__ __forceinline__ __nv_bfloat16 from_fp32<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

// Four 8-lane subgroups process the same shared token for four query heads.
// The first subgroup loads K/V and broadcasts packed values across its warp.
template<typename T, int Segments, bool Tiled>
__global__ void group_warp(const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v,
                           const int32_t* __restrict__ shared, const int32_t* __restrict__ ex,
                           const int32_t* __restrict__ sc, const int32_t* __restrict__ ec,
                           T* __restrict__ out, float* __restrict__ partial,
                           int heads, int groups, int length, int sw, int ew, int max_parts, int chunk) {
  const int batch = blockIdx.x, group = blockIdx.y, part = blockIdx.z;
  const int lane = threadIdx.x & 7, h = (threadIdx.x & 31) >> 3, segment = threadIdx.x >> 5;
  const int64_t qb = (static_cast<int64_t>(batch) * heads + group * 4 + h) * 128;
  const int64_t kb = (static_cast<int64_t>(batch) * groups + group) * length * 128;
  const int64_t sb = (static_cast<int64_t>(batch) * groups + group) * sw;
  const int64_t eb = (static_cast<int64_t>(batch) * heads + group * 4 + h) * ew;
  int max_count = sc[batch * groups + group];
  #pragma unroll
  for (int i = 0; i < 4; ++i) max_count = max(max_count, ec[batch * heads + group * 4 + i]);
  const int parts = min(max_parts, max(1, (max_count + chunk - 1) / chunk));
  if (part >= parts) return;
  const uint4 q0 = *reinterpret_cast<const uint4*>(q + qb + lane * 16);
  const uint4 q1 = *reinterpret_cast<const uint4*>(q + qb + lane * 16 + 8);
  uint4 qraw[2] = {q0, q1};
  const T* qvalues = reinterpret_cast<const T*>(qraw);
  float query[16], acc[16];
  #pragma unroll
  for (int i = 0; i < 16; ++i) { query[i] = fp32<T>(qvalues[i]); acc[i] = 0; }
  float maximum = -INFINITY, sum = 0;
  const int ns = sc[batch * groups + group], span = (ns + parts - 1) / parts;
  const int start = part * span, end = min(ns, start + span);
  if (Tiled) {
    constexpr int Tile = 4;
    for (int pos = start + segment; pos < end; pos += Segments * Tile) {
      float scores[Tile]; int tokens[Tile]; float tile_max = -INFINITY;
      #pragma unroll
      for (int t = 0; t < Tile; ++t) {
        const int at = pos + t * Segments;
        const int token = at < end ? shared[sb + at] : -1;
        tokens[t] = token;
        scores[t] = -INFINITY;
        if (token < 0 || token >= length) continue;
        const int64_t base = kb + static_cast<int64_t>(token) * 128 + lane * 16;
        uint4 kr[2] = {};
        if (h == 0) {
          kr[0] = *reinterpret_cast<const uint4*>(k + base);
          kr[1] = *reinterpret_cast<const uint4*>(k + base + 8);
        }
        unsigned* packed = reinterpret_cast<unsigned*>(kr);
        #pragma unroll
        for (int i = 0; i < 8; ++i) packed[i] = __shfl_sync(0xffffffff, packed[i], lane);
        const T* kval = reinterpret_cast<const T*>(kr);
        float dot = 0;
        #pragma unroll
        for (int i = 0; i < 16; ++i) dot += query[i] * fp32<T>(kval[i]);
        #pragma unroll
        for (int d = 4; d; d >>= 1) dot += __shfl_down_sync(0xffffffff, dot, d, 8);
        scores[t] = __shfl_sync(0xffffffff, dot, 0, 8) * 0.08838834764831845f;
        tile_max = fmaxf(tile_max, scores[t]);
      }
      if (tile_max == -INFINITY) continue;
      const float next = fmaxf(maximum, tile_max);
      const float scale = sum > 0 ? __expf(maximum - next) : 0;
      #pragma unroll
      for (int i = 0; i < 16; ++i) acc[i] *= scale;
      sum *= scale;
      #pragma unroll
      for (int t = 0; t < Tile; ++t) {
        const int token = tokens[t];
        if (token < 0 || token >= length) continue;
        const int64_t base = kb + static_cast<int64_t>(token) * 128 + lane * 16;
        uint4 vr[2] = {};
        if (h == 0) {
          vr[0] = *reinterpret_cast<const uint4*>(v + base);
          vr[1] = *reinterpret_cast<const uint4*>(v + base + 8);
        }
        unsigned* packed = reinterpret_cast<unsigned*>(vr);
        #pragma unroll
        for (int i = 0; i < 8; ++i) packed[i] = __shfl_sync(0xffffffff, packed[i], lane);
        const T* vval = reinterpret_cast<const T*>(vr);
        const float weight = __expf(scores[t] - next);
        #pragma unroll
        for (int i = 0; i < 16; ++i) acc[i] += weight * fp32<T>(vval[i]);
        sum += weight;
      }
      maximum = next;
    }
  } else for (int pos = start + segment; pos < end; pos += Segments) {
    const int token = shared[sb + pos];
    if (token < 0 || token >= length) continue;
    const int64_t base = kb + static_cast<int64_t>(token) * 128 + lane * 16;
    uint4 kr[2] = {}, vr[2] = {};
    if (h == 0) {
      kr[0] = *reinterpret_cast<const uint4*>(k + base);
      kr[1] = *reinterpret_cast<const uint4*>(k + base + 8);
      vr[0] = *reinterpret_cast<const uint4*>(v + base);
      vr[1] = *reinterpret_cast<const uint4*>(v + base + 8);
    }
    unsigned* kp = reinterpret_cast<unsigned*>(kr);
    unsigned* vp = reinterpret_cast<unsigned*>(vr);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      kp[i] = __shfl_sync(0xffffffff, kp[i], lane);
      vp[i] = __shfl_sync(0xffffffff, vp[i], lane);
    }
    const T* kval = reinterpret_cast<const T*>(kr);
    const T* vval = reinterpret_cast<const T*>(vr);
    float dot = 0;
    #pragma unroll
    for (int i = 0; i < 16; ++i) dot += query[i] * fp32<T>(kval[i]);
    #pragma unroll
    for (int d = 4; d; d >>= 1) dot += __shfl_down_sync(0xffffffff, dot, d, 8);
    dot = __shfl_sync(0xffffffff, dot, 0, 8) * 0.08838834764831845f;
    const float next = fmaxf(maximum, dot), a = __expf(maximum - next), b = __expf(dot - next);
    #pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = acc[i] * a + b * fp32<T>(vval[i]);
    sum = sum * a + b;
    maximum = next;
  }
  const int ne = ec[batch * heads + group * 4 + h], espan = (ne + parts - 1) / parts;
  const int estart = part * espan, eend = min(ne, estart + espan);
  for (int pos = estart + segment; pos < eend; pos += Segments) {
    const int token = ex[eb + pos];
    if (token < 0 || token >= length) continue;
    const int64_t base = kb + static_cast<int64_t>(token) * 128 + lane * 16;
    uint4 kr[2] = {*reinterpret_cast<const uint4*>(k + base), *reinterpret_cast<const uint4*>(k + base + 8)};
    uint4 vr[2] = {*reinterpret_cast<const uint4*>(v + base), *reinterpret_cast<const uint4*>(v + base + 8)};
    const T* kval = reinterpret_cast<const T*>(kr);
    const T* vval = reinterpret_cast<const T*>(vr);
    float dot = 0;
    #pragma unroll
    for (int i = 0; i < 16; ++i) dot += query[i] * fp32<T>(kval[i]);
    const unsigned active = __activemask();
    #pragma unroll
    for (int d = 4; d; d >>= 1) dot += __shfl_down_sync(active, dot, d, 8);
    dot = __shfl_sync(active, dot, 0, 8) * 0.08838834764831845f;
    const float next = fmaxf(maximum, dot), a = __expf(maximum - next), b = __expf(dot - next);
    #pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = acc[i] * a + b * fp32<T>(vval[i]);
    sum = sum * a + b;
    maximum = next;
  }
  __shared__ float states[Segments][4][128], meta[Segments][4][2];
  #pragma unroll
  for (int i = 0; i < 16; ++i) states[segment][h][lane * 16 + i] = acc[i];
  if (!lane) { meta[segment][h][0] = maximum; meta[segment][h][1] = sum; }
  __syncthreads();
  if (segment == 0) {
    float maxval = -INFINITY, denom = 0, result[16] = {0};
    #pragma unroll
    for (int s = 0; s < Segments; ++s) maxval = fmaxf(maxval, meta[s][h][0]);
    #pragma unroll
    for (int s = 0; s < Segments; ++s) {
      if (meta[s][h][1] == 0) continue;
      const float weight = __expf(meta[s][h][0] - maxval);
      denom += meta[s][h][1] * weight;
      #pragma unroll
      for (int i = 0; i < 16; ++i) result[i] += states[s][h][lane * 16 + i] * weight;
    }
    if (parts == 1) {
      uint4 raw[2]; T* dst = reinterpret_cast<T*>(raw);
      #pragma unroll
      for (int i = 0; i < 16; ++i) dst[i] = from_fp32<T>(result[i] / denom);
      *reinterpret_cast<uint4*>(out + qb + lane * 16) = raw[0];
      *reinterpret_cast<uint4*>(out + qb + lane * 16 + 8) = raw[1];
    } else {
      const int64_t base = (((static_cast<int64_t>(batch) * groups + group) * max_parts + part) * 4 + h) * 130;
      #pragma unroll
      for (int i = 0; i < 16; ++i) partial[base + lane * 16 + i] = result[i];
      if (!lane) { partial[base + 128] = maxval; partial[base + 129] = denom; }
    }
  }
}

template<typename T>
__global__ void merge(const float* __restrict__ partial, T* __restrict__ out, const int32_t* sc,
                      const int32_t* ec, int heads, int groups, int max_parts, int chunk) {
  const int batch = blockIdx.x, head = blockIdx.y, group = head / 4, lane = threadIdx.x;
  int count = sc[batch * groups + group];
  #pragma unroll
  for (int h = 0; h < 4; ++h) count = max(count, ec[batch * heads + group * 4 + h]);
  const int parts = min(max_parts, max(1, (count + chunk - 1) / chunk));
  if (parts == 1) return;
  const int64_t base = (static_cast<int64_t>(batch) * groups + group) * max_parts * 4 * 130;
  float maxval = -INFINITY, denom = 0, result[8] = {0};
  for (int p = 0; p < parts; ++p) maxval = fmaxf(maxval, partial[base + (p * 4 + head % 4) * 130 + 128]);
  for (int p = 0; p < parts; ++p) {
    const int64_t offset = base + (p * 4 + head % 4) * 130;
    const float psum = partial[offset + 129];
    if (psum == 0) continue;
    const float weight = __expf(partial[offset + 128] - maxval);
    denom += psum * weight;
    #pragma unroll
    for (int i = 0; i < 8; ++i) result[i] += partial[offset + lane * 8 + i] * weight;
  }
  uint4 raw; T* dst = reinterpret_cast<T*>(&raw);
  #pragma unroll
  for (int i = 0; i < 8; ++i) dst[i] = from_fp32<T>(result[i] / denom);
  *reinterpret_cast<uint4*>(out + (static_cast<int64_t>(batch) * heads + head) * 128 + lane * 8) = raw;
}

template<typename T, bool Tiled> void run(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor shared,
    torch::Tensor ex, torch::Tensor sc, torch::Tensor ec, torch::Tensor out, int segments, int parts, int chunk) {
  auto partial = torch::empty({q.size(0), k.size(1), parts, 4, 130}, q.options().dtype(at::kFloat));
  const auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(q.size(0), k.size(1), parts);
  #define RUN(N) group_warp<T,N,Tiled><<<grid, N * 32, 0, stream>>>(reinterpret_cast<const T*>(q.data_ptr()), reinterpret_cast<const T*>(k.data_ptr()), reinterpret_cast<const T*>(v.data_ptr()), shared.data_ptr<int32_t>(), ex.data_ptr<int32_t>(), sc.data_ptr<int32_t>(), ec.data_ptr<int32_t>(), reinterpret_cast<T*>(out.data_ptr()), partial.data_ptr<float>(), q.size(1), k.size(1), k.size(2), shared.size(2), ex.size(2), parts, chunk)
  if (segments == 4) { RUN(4); } else { RUN(8); }
  #undef RUN
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  merge<T><<<dim3(q.size(0), q.size(1)), 16, 0, stream>>>(partial.data_ptr<float>(), reinterpret_cast<T*>(out.data_ptr()), sc.data_ptr<int32_t>(), ec.data_ptr<int32_t>(), q.size(1), k.size(1), parts, chunk);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor decode_impl(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor shared,
                          torch::Tensor ex, torch::Tensor sc, torch::Tensor ec, int segments, int parts, int chunk, bool tiled) {
  for (const auto& x : {q,k,v,shared,ex,sc,ec}) TORCH_CHECK(x.is_cuda() && x.device() == q.device() && x.is_contiguous(), "contiguous tensors on one CUDA device required");
  TORCH_CHECK((q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16) && q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(), "FP16 or BF16 required");
  for (const auto& x : {shared,ex,sc,ec}) TORCH_CHECK(x.scalar_type() == at::kInt, "int32 metadata required");
  TORCH_CHECK(q.dim() == 3 && k.dim() == 4 && k.sizes() == v.sizes() && q.size(0) == k.size(0) && q.size(1) == 4*k.size(1) && q.size(2) == 128 && k.size(3) == 128, "invalid Q/K/V shape");
  TORCH_CHECK(shared.dim() == 3 && shared.size(0) == q.size(0) && shared.size(1) == k.size(1) && ex.dim() == 3 && ex.size(0) == q.size(0) && ex.size(1) == q.size(1), "invalid support shape");
  TORCH_CHECK(sc.dim() == 2 && sc.size(0) == q.size(0) && sc.size(1) == k.size(1) && ec.dim() == 2 && ec.size(0) == q.size(0) && ec.size(1) == q.size(1), "invalid count shape");
  TORCH_CHECK(k.size(2) > 0 && shared.size(2) + ex.size(2) > 0 && (segments == 4 || segments == 8), "invalid size or segments");
  TORCH_CHECK((parts == 2 || parts == 4 || parts == 8 || parts == 16 || parts == 32) && chunk > 0, "invalid partition configuration");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(k.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(v.data_ptr()) % 16 == 0, "16-byte alignment required");
  const c10::cuda::CUDAGuard guard(q.device());
  auto out = torch::empty_like(q);
  if (q.scalar_type() == at::kHalf) {
    if (tiled) run<half, true>(q,k,v,shared,ex,sc,ec,out,segments,parts,chunk);
    else run<half, false>(q,k,v,shared,ex,sc,ec,out,segments,parts,chunk);
  } else {
    if (tiled) run<__nv_bfloat16, true>(q,k,v,shared,ex,sc,ec,out,segments,parts,chunk);
    else run<__nv_bfloat16, false>(q,k,v,shared,ex,sc,ec,out,segments,parts,chunk);
  }
  return out;
}

torch::Tensor decode(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor shared,
                     torch::Tensor ex, torch::Tensor sc, torch::Tensor ec, int segments, int parts, int chunk) {
  return decode_impl(q,k,v,shared,ex,sc,ec,segments,parts,chunk,false);
}

torch::Tensor decode_tiled(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor shared,
                           torch::Tensor ex, torch::Tensor sc, torch::Tensor ec, int segments, int parts, int chunk) {
  return decode_impl(q,k,v,shared,ex,sc,ec,segments,parts,chunk,true);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("decode", &decode); m.def("decode_tiled", &decode_tiled); }
