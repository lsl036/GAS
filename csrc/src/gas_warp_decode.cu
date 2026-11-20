#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// A warp handles one query head. Four warps in a block address the same KV
// group, so the shared indices and cache lines can be reused by the hardware.
__global__ void gas_warp_kernel(const half* __restrict__ q, const half* __restrict__ k,
                                const half* __restrict__ v, const int32_t* __restrict__ shared,
                                const int32_t* __restrict__ exceptions, half* __restrict__ out,
                                int heads, int kv_heads, int length, int shared_width, int ex_width) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int batch = blockIdx.x;
  const int kvhead = blockIdx.y;
  const int head = kvhead * 4 + warp;
  const int64_t qbase = (static_cast<int64_t>(batch) * heads + head) * 128;
  const int64_t kvbase = (static_cast<int64_t>(batch) * kv_heads + kvhead) * length * 128;
  const int64_t sbase = (static_cast<int64_t>(batch) * kv_heads + kvhead) * shared_width;
  const int64_t ebase = (static_cast<int64_t>(batch) * heads + head) * ex_width;
  const float q0 = __half2float(q[qbase + lane]);
  const float q1 = __half2float(q[qbase + lane + 32]);
  const float q2 = __half2float(q[qbase + lane + 64]);
  const float q3 = __half2float(q[qbase + lane + 96]);
  float o0 = 0, o1 = 0, o2 = 0, o3 = 0, maxval = -INFINITY, sum = 0;
  for (int i = 0; i < shared_width + ex_width; ++i) {
    const int token = i < shared_width ? shared[sbase + i] : exceptions[ebase + i - shared_width];
    if (token < 0 || token >= length) continue;
    const int64_t base = kvbase + static_cast<int64_t>(token) * 128;
    float dot = q0 * __half2float(k[base + lane]) + q1 * __half2float(k[base + lane + 32]) +
                q2 * __half2float(k[base + lane + 64]) + q3 * __half2float(k[base + lane + 96]);
    for (int d = 16; d; d >>= 1) dot += __shfl_down_sync(0xffffffff, dot, d);
    dot = __shfl_sync(0xffffffff, dot, 0) * 0.08838834764831845f;
    const float next_max = fmaxf(maxval, dot);
    const float old_weight = __expf(maxval - next_max);
    const float weight = __expf(dot - next_max);
    sum = sum * old_weight + weight;
    o0 = o0 * old_weight + weight * __half2float(v[base + lane]);
    o1 = o1 * old_weight + weight * __half2float(v[base + lane + 32]);
    o2 = o2 * old_weight + weight * __half2float(v[base + lane + 64]);
    o3 = o3 * old_weight + weight * __half2float(v[base + lane + 96]);
    maxval = next_max;
  }
  out[qbase + lane] = __float2half(o0 / sum);
  out[qbase + lane + 32] = __float2half(o1 / sum);
  out[qbase + lane + 64] = __float2half(o2 / sum);
  out[qbase + lane + 96] = __float2half(o3 / sum);
}

// Four warps cooperate on one head's QK, then each thread produces one output
// dimension. Heads still address the same group indices but reuse KV via cache.
__global__ void gas_head_parallel_kernel(const half* __restrict__ q, const half* __restrict__ k,
                                         const half* __restrict__ v, const int32_t* __restrict__ shared,
                                         const int32_t* __restrict__ exceptions, half* __restrict__ out,
                                         int heads, int kv_heads, int length, int sw, int ew) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int batch = blockIdx.x, head = blockIdx.y, kvhead = head / 4;
  const int64_t qb = (static_cast<int64_t>(batch) * heads + head) * 128;
  const int64_t kb = (static_cast<int64_t>(batch) * kv_heads + kvhead) * length * 128;
  const int64_t sb = (static_cast<int64_t>(batch) * kv_heads + kvhead) * sw;
  const int64_t eb = (static_cast<int64_t>(batch) * heads + head) * ew;
  const float q0 = __half2float(q[qb + lane]);
  const float q1 = __half2float(q[qb + lane + 32]);
  const float q2 = __half2float(q[qb + lane + 64]);
  const float q3 = __half2float(q[qb + lane + 96]);
  extern __shared__ float score[];
  __shared__ float reduce[128];
  for (int i = warp; i < sw + ew; i += 4) {
    const int token = i < sw ? shared[sb + i] : exceptions[eb + i - sw];
    float dot = 0;
    if (token >= 0 && token < length) {
      const int64_t base = kb + static_cast<int64_t>(token) * 128;
      dot = q0 * __half2float(k[base + lane]) + q1 * __half2float(k[base + lane + 32]) +
            q2 * __half2float(k[base + lane + 64]) + q3 * __half2float(k[base + lane + 96]);
    }
    for (int d = 16; d; d >>= 1) dot += __shfl_down_sync(0xffffffff, dot, d);
    if (!lane) score[i] = token >= 0 && token < length ? dot * 0.08838834764831845f : -INFINITY;
  }
  __syncthreads();
  float maximum = -INFINITY;
  for (int i = tid; i < sw + ew; i += 128) maximum = fmaxf(maximum, score[i]);
  reduce[tid] = maximum;
  __syncthreads();
  for (int step = 64; step; step >>= 1) {
    if (tid < step) reduce[tid] = fmaxf(reduce[tid], reduce[tid + step]);
    __syncthreads();
  }
  maximum = reduce[0];
  __syncthreads();
  float sum = 0;
  for (int i = tid; i < sw + ew; i += 128) {
    float w = __expf(score[i] - maximum);
    score[i] = w;
    sum += w;
  }
  reduce[tid] = sum;
  __syncthreads();
  for (int step = 64; step; step >>= 1) {
    if (tid < step) reduce[tid] += reduce[tid + step];
    __syncthreads();
  }
  const float invsum = 1.0f / reduce[0];
  float result = 0;
  for (int i = 0; i < sw + ew; ++i) {
    const int token = i < sw ? shared[sb + i] : exceptions[eb + i - sw];
    if (token >= 0 && token < length)
      result += score[i] * __half2float(v[kb + static_cast<int64_t>(token) * 128 + tid]);
  }
  out[qb + tid] = __float2half(result * invsum);
}

// Split the support over independent CTAs; each emits an unnormalized online
// softmax state, which is merged exactly in the second kernel.
template<bool StageShared>
__global__ void gas_split_kernel(const half* __restrict__ q, const half* __restrict__ k,
                                 const half* __restrict__ v, const int32_t* __restrict__ shared,
                                 const int32_t* __restrict__ exceptions, float* __restrict__ partial,
                                 int heads, int kv_heads, int length, int sw, int ew, int chunks, int tile) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int batch = blockIdx.x, kvhead = blockIdx.y, chunk = blockIdx.z;
  const int head = kvhead * 4 + warp;
  const int64_t qb = (static_cast<int64_t>(batch) * heads + head) * 128;
  const int64_t kb = (static_cast<int64_t>(batch) * kv_heads + kvhead) * length * 128;
  const int64_t sb = (static_cast<int64_t>(batch) * kv_heads + kvhead) * sw;
  const int64_t eb = (static_cast<int64_t>(batch) * heads + head) * ew;
  const int64_t pb = ((static_cast<int64_t>(batch) * heads + head) * chunks + chunk) * 130;
  const float q0 = __half2float(q[qb + lane]), q1 = __half2float(q[qb + lane + 32]);
  const float q2 = __half2float(q[qb + lane + 64]), q3 = __half2float(q[qb + lane + 96]);
  float o0 = 0, o1 = 0, o2 = 0, o3 = 0, m = -INFINITY, sum = 0;
  const int end = min((chunk + 1) * tile, sw + ew);
  extern __shared__ half staged[];
  if (StageShared) {
    for (int j = threadIdx.x; j < tile * 128; j += blockDim.x) {
      const int i = chunk * tile + j / 128;
      const int token = i < sw ? shared[sb + i] : -1;
      const bool valid = token >= 0 && token < length;
      const int64_t address = kb + static_cast<int64_t>(valid ? token : 0) * 128 + j % 128;
      staged[j] = valid ? k[address] : __float2half(0);
      staged[tile * 128 + j] = valid ? v[address] : __float2half(0);
    }
    __syncthreads();
  }
  for (int i = chunk * tile; i < end; ++i) {
    const int token = i < sw ? shared[sb + i] : exceptions[eb + i - sw];
    if (token < 0 || token >= length) continue;
    const int64_t base = kb + static_cast<int64_t>(token) * 128;
    const bool use_staged = StageShared && i < sw;
    const int local = (i - chunk * tile) * 128;
    const half* kp = use_staged ? staged + local : k + base;
    const half* vp = use_staged ? staged + tile * 128 + local : v + base;
    float dot = q0 * __half2float(kp[lane]) + q1 * __half2float(kp[lane + 32]) +
                q2 * __half2float(kp[lane + 64]) + q3 * __half2float(kp[lane + 96]);
    for (int d = 16; d; d >>= 1) dot += __shfl_down_sync(0xffffffff, dot, d);
    dot = __shfl_sync(0xffffffff, dot, 0) * 0.08838834764831845f;
    const float next = fmaxf(m, dot), a = __expf(m - next), b = __expf(dot - next);
    sum = sum * a + b;
    o0 = o0 * a + b * __half2float(vp[lane]);
    o1 = o1 * a + b * __half2float(vp[lane + 32]);
    o2 = o2 * a + b * __half2float(vp[lane + 64]);
    o3 = o3 * a + b * __half2float(vp[lane + 96]);
    m = next;
  }
  partial[pb + lane] = o0; partial[pb + lane + 32] = o1;
  partial[pb + lane + 64] = o2; partial[pb + lane + 96] = o3;
  if (!lane) { partial[pb + 128] = m; partial[pb + 129] = sum; }
}

__global__ void gas_merge_kernel(const float* __restrict__ partial, half* __restrict__ out,
                                 int heads, int chunks) {
  const int lane = threadIdx.x, batch = blockIdx.x, head = blockIdx.y;
  const int64_t base = (static_cast<int64_t>(batch) * heads + head) * chunks * 130;
  float maximum = -INFINITY;
  for (int i = 0; i < chunks; ++i) maximum = fmaxf(maximum, partial[base + i * 130 + 128]);
  float denominator = 0, o0 = 0, o1 = 0, o2 = 0, o3 = 0;
  for (int i = 0; i < chunks; ++i) {
    const int64_t pb = base + i * 130;
    const float sum = partial[pb + 129];
    if (sum == 0) continue;
    const float scale = __expf(partial[pb + 128] - maximum);
    denominator += sum * scale;
    o0 += partial[pb + lane] * scale; o1 += partial[pb + lane + 32] * scale;
    o2 += partial[pb + lane + 64] * scale; o3 += partial[pb + lane + 96] * scale;
  }
  const int64_t qb = (static_cast<int64_t>(batch) * heads + head) * 128;
  out[qb + lane] = __float2half(o0 / denominator); out[qb + lane + 32] = __float2half(o1 / denominator);
  out[qb + lane + 64] = __float2half(o2 / denominator); out[qb + lane + 96] = __float2half(o3 / denominator);
}

// One CTA per head stages a tile, computes parallel QK, and updates an online
// softmax/PV state. KV sharing across head CTAs is through the cache only.
__global__ void gas_tiled_head_kernel(const half* __restrict__ q, const half* __restrict__ k,
                                      const half* __restrict__ v, const int32_t* __restrict__ shared,
                                      const int32_t* __restrict__ exceptions, half* __restrict__ out,
                                      int heads, int kv_heads, int length, int sw, int ew) {
  constexpr int tile = 16;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int batch = blockIdx.x, head = blockIdx.y, kvhead = head / 4;
  const int64_t qb = (static_cast<int64_t>(batch) * heads + head) * 128;
  const int64_t kb = (static_cast<int64_t>(batch) * kv_heads + kvhead) * length * 128;
  const int64_t sb = (static_cast<int64_t>(batch) * kv_heads + kvhead) * sw;
  const int64_t eb = (static_cast<int64_t>(batch) * heads + head) * ew;
  __shared__ half sk[tile * 128], sv[tile * 128];
  __shared__ float scores[tile], weights[tile], running_max, running_sum, alpha;
  if (tid == 0) { running_max = -INFINITY; running_sum = 0; }
  const float q0 = __half2float(q[qb + lane]), q1 = __half2float(q[qb + lane + 32]);
  const float q2 = __half2float(q[qb + lane + 64]), q3 = __half2float(q[qb + lane + 96]);
  float acc = 0;
  for (int start = 0; start < sw + ew; start += tile) {
    for (int j = tid; j < tile * 128; j += 128) {
      const int pos = start + j / 128;
      const int token = pos < sw ? shared[sb + pos] : (pos < sw + ew ? exceptions[eb + pos - sw] : -1);
      const bool valid = token >= 0 && token < length;
      const int64_t address = kb + static_cast<int64_t>(valid ? token : 0) * 128 + j % 128;
      sk[j] = valid ? k[address] : __float2half(0);
      sv[j] = valid ? v[address] : __float2half(0);
    }
    __syncthreads();
    for (int t = warp; t < tile; t += 4) {
      const int pos = start + t;
      const int token = pos < sw ? shared[sb + pos] : (pos < sw + ew ? exceptions[eb + pos - sw] : -1);
      const int base = t * 128 + lane;
      float dot = q0 * __half2float(sk[base]) + q1 * __half2float(sk[base + 32]) +
                  q2 * __half2float(sk[base + 64]) + q3 * __half2float(sk[base + 96]);
      for (int d = 16; d; d >>= 1) dot += __shfl_down_sync(0xffffffff, dot, d);
      if (!lane) scores[t] = token >= 0 && token < length ? dot * 0.08838834764831845f : -INFINITY;
    }
    __syncthreads();
    if (warp == 0) {
      float s = lane < tile ? scores[lane] : -INFINITY;
      float tile_max = s;
      for (int d = 16; d; d >>= 1) tile_max = fmaxf(tile_max, __shfl_down_sync(0xffffffff, tile_max, d));
      tile_max = __shfl_sync(0xffffffff, tile_max, 0);
      const float next = fmaxf(running_max, tile_max);
      const bool has_value = next != -INFINITY;
      const float a = has_value ? __expf(running_max - next) : 1.0f;
      const float w = has_value ? __expf(s - next) : 0.0f;
      float tile_sum = w;
      for (int d = 16; d; d >>= 1) tile_sum += __shfl_down_sync(0xffffffff, tile_sum, d);
      if (lane < tile) weights[lane] = w;
      if (!lane) { alpha = a; running_sum = running_sum * a + tile_sum; running_max = next; }
    }
    __syncthreads();
    acc *= alpha;
    #pragma unroll
    for (int t = 0; t < tile; ++t) acc += weights[t] * __half2float(sv[t * 128 + tid]);
    __syncthreads();
  }
  out[qb + tid] = __float2half(acc / running_sum);
}

// Token-parallel online softmax: independent warp states are reduced within
// the CTA. Each CTA owns one query head and consumes its exact GAS support.
template<int Warps>
__global__ void gas_token_parallel_kernel(const half* __restrict__ q, const half* __restrict__ k,
                                         const half* __restrict__ v, const int32_t* __restrict__ shared,
                                         const int32_t* __restrict__ exceptions, half* __restrict__ out,
                                         int heads, int kv_heads, int length, int sw, int ew) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int batch = blockIdx.x, head = blockIdx.y, kvhead = head / 4;
  const int64_t qb = (static_cast<int64_t>(batch) * heads + head) * 128;
  const int64_t kb = (static_cast<int64_t>(batch) * kv_heads + kvhead) * length * 128;
  const int64_t sb = (static_cast<int64_t>(batch) * kv_heads + kvhead) * sw;
  const int64_t eb = (static_cast<int64_t>(batch) * heads + head) * ew;
  const float q0 = __half2float(q[qb + lane]), q1 = __half2float(q[qb + lane + 32]);
  const float q2 = __half2float(q[qb + lane + 64]), q3 = __half2float(q[qb + lane + 96]);
  float o0 = 0, o1 = 0, o2 = 0, o3 = 0, m = -INFINITY, sum = 0;
  for (int i = warp; i < sw + ew; i += Warps) {
    const int token = i < sw ? shared[sb + i] : exceptions[eb + i - sw];
    if (token < 0 || token >= length) continue;
    const int64_t base = kb + static_cast<int64_t>(token) * 128;
    float dot = q0 * __half2float(k[base + lane]) + q1 * __half2float(k[base + lane + 32]) +
                q2 * __half2float(k[base + lane + 64]) + q3 * __half2float(k[base + lane + 96]);
    for (int d = 16; d; d >>= 1) dot += __shfl_down_sync(0xffffffff, dot, d);
    dot = __shfl_sync(0xffffffff, dot, 0) * 0.08838834764831845f;
    const float next = fmaxf(m, dot), a = __expf(m - next), b = __expf(dot - next);
    sum = sum * a + b;
    o0 = o0 * a + b * __half2float(v[base + lane]);
    o1 = o1 * a + b * __half2float(v[base + lane + 32]);
    o2 = o2 * a + b * __half2float(v[base + lane + 64]);
    o3 = o3 * a + b * __half2float(v[base + lane + 96]);
    m = next;
  }
  __shared__ float states[Warps][130];
  states[warp][lane] = o0; states[warp][lane + 32] = o1;
  states[warp][lane + 64] = o2; states[warp][lane + 96] = o3;
  if (!lane) { states[warp][128] = m; states[warp][129] = sum; }
  __syncthreads();
  if (warp == 0) {
    float maximum = -INFINITY;
    #pragma unroll
    for (int w = 0; w < Warps; ++w) maximum = fmaxf(maximum, states[w][128]);
    float denominator = 0, r0 = 0, r1 = 0, r2 = 0, r3 = 0;
    #pragma unroll
    for (int w = 0; w < Warps; ++w) {
      if (states[w][129] == 0) continue;
      const float weight = __expf(states[w][128] - maximum);
      denominator += states[w][129] * weight;
      r0 += states[w][lane] * weight; r1 += states[w][lane + 32] * weight;
      r2 += states[w][lane + 64] * weight; r3 += states[w][lane + 96] * weight;
    }
    out[qb + lane] = __float2half(r0 / denominator); out[qb + lane + 32] = __float2half(r1 / denominator);
    out[qb + lane + 64] = __float2half(r2 / denominator); out[qb + lane + 96] = __float2half(r3 / denominator);
  }
}

// A 16-thread subgroup owns one token and each lane loads eight contiguous
// elements. Subgroups process independent token streams and merge online states.
template<int Segments>
__global__ void gas_vector16_kernel(const half* __restrict__ q, const half* __restrict__ k,
                                    const half* __restrict__ v, const int32_t* __restrict__ shared,
                                    const int32_t* __restrict__ exceptions, half* __restrict__ out,
                                    int heads, int kv_heads, int length, int sw, int ew) {
  const int lane = threadIdx.x & 15, segment = threadIdx.x >> 4;
  const int batch = blockIdx.x, head = blockIdx.y, kvhead = head / 4;
  const int64_t qb = (static_cast<int64_t>(batch) * heads + head) * 128;
  const int64_t kb = (static_cast<int64_t>(batch) * kv_heads + kvhead) * length * 128;
  const int64_t sb = (static_cast<int64_t>(batch) * kv_heads + kvhead) * sw;
  const int64_t eb = (static_cast<int64_t>(batch) * heads + head) * ew;
  const uint4 qraw = *reinterpret_cast<const uint4*>(q + qb + lane * 8);
  const half* qhalf = reinterpret_cast<const half*>(&qraw);
  float qvec[8], acc[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) { qvec[i] = __half2float(qhalf[i]); acc[i] = 0; }
  float m = -INFINITY, sum = 0;
  for (int pos = segment; pos < sw + ew; pos += Segments) {
    const int token = pos < sw ? shared[sb + pos] : exceptions[eb + pos - sw];
    if (token < 0 || token >= length) continue;
    const int64_t base = kb + static_cast<int64_t>(token) * 128 + lane * 8;
    const uint4 kraw = *reinterpret_cast<const uint4*>(k + base);
    const half* khalf = reinterpret_cast<const half*>(&kraw);
    float dot = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) dot += qvec[i] * __half2float(khalf[i]);
    const unsigned active = __activemask();
    for (int d = 8; d; d >>= 1) dot += __shfl_down_sync(active, dot, d, 16);
    dot = __shfl_sync(active, dot, 0, 16) * 0.08838834764831845f;
    const float next = fmaxf(m, dot), a = __expf(m - next), b = __expf(dot - next);
    const uint4 vraw = *reinterpret_cast<const uint4*>(v + base);
    const half* vhalf = reinterpret_cast<const half*>(&vraw);
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = acc[i] * a + b * __half2float(vhalf[i]);
    sum = sum * a + b;
    m = next;
  }
  __shared__ float states[Segments][8][16];
  __shared__ float meta[Segments][2];
  #pragma unroll
  for (int i = 0; i < 8; ++i) states[segment][i][lane] = acc[i];
  if (!lane) { meta[segment][0] = m; meta[segment][1] = sum; }
  __syncthreads();
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
    half* h = reinterpret_cast<half*>(&raw);
    #pragma unroll
    for (int i = 0; i < 8; ++i) h[i] = __float2half(result[i] / denominator);
    *reinterpret_cast<uint4*>(out + qb + lane * 8) = raw;
  }
}

torch::Tensor gas_decode(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                         torch::Tensor shared, torch::Tensor exceptions, int mode, int tile) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && shared.is_cuda() && exceptions.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(q.device() == k.device() && q.device() == v.device() && q.device() == shared.device() && q.device() == exceptions.device(), "device mismatch");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && shared.is_contiguous() && exceptions.is_contiguous(), "contiguous tensors required");
  TORCH_CHECK(q.scalar_type() == at::kHalf && k.scalar_type() == at::kHalf && v.scalar_type() == at::kHalf, "FP16 required");
  TORCH_CHECK(shared.scalar_type() == at::kInt && exceptions.scalar_type() == at::kInt, "int32 indices required");
  TORCH_CHECK(q.dim() == 3 && k.dim() == 4 && v.sizes() == k.sizes(), "invalid Q/K/V shape");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(1) == k.size(1) * 4 && q.size(2) == 128 && k.size(3) == 128, "expected group=4 and dim=128");
  TORCH_CHECK(shared.dim() == 3 && shared.size(0) == q.size(0) && shared.size(1) == k.size(1) && exceptions.dim() == 3 && exceptions.size(0) == q.size(0) && exceptions.size(1) == q.size(1), "invalid index shape");
  TORCH_CHECK(mode >= 0 && mode <= 11, "invalid mode");
  TORCH_CHECK(k.size(2) > 0 && shared.size(2) + exceptions.size(2) > 0, "empty sequence or support");
  if (mode >= 8) {
    TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(k.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(v.data_ptr()) % 16 == 0, "vector inputs must be 16-byte aligned");
  }
  const c10::cuda::CUDAGuard guard(q.device());
  auto out = torch::empty_like(q);
  auto qptr = reinterpret_cast<const half*>(q.data_ptr<at::Half>());
  auto kptr = reinterpret_cast<const half*>(k.data_ptr<at::Half>());
  auto vptr = reinterpret_cast<const half*>(v.data_ptr<at::Half>());
  auto optr = reinterpret_cast<half*>(out.data_ptr<at::Half>());
  if (mode >= 8) {
    const dim3 grid(q.size(0), q.size(1));
    const auto stream = at::cuda::getCurrentCUDAStream();
    if (mode == 8)
      gas_vector16_kernel<4><<<grid, 64, 0, stream>>>(qptr, kptr, vptr, shared.data_ptr<int32_t>(), exceptions.data_ptr<int32_t>(), optr, q.size(1), k.size(1), k.size(2), shared.size(2), exceptions.size(2));
    else if (mode == 9)
      gas_vector16_kernel<8><<<grid, 128, 0, stream>>>(qptr, kptr, vptr, shared.data_ptr<int32_t>(), exceptions.data_ptr<int32_t>(), optr, q.size(1), k.size(1), k.size(2), shared.size(2), exceptions.size(2));
    else if (mode == 10)
      gas_vector16_kernel<16><<<grid, 256, 0, stream>>>(qptr, kptr, vptr, shared.data_ptr<int32_t>(), exceptions.data_ptr<int32_t>(), optr, q.size(1), k.size(1), k.size(2), shared.size(2), exceptions.size(2));
    else
      gas_vector16_kernel<32><<<grid, 512, 0, stream>>>(qptr, kptr, vptr, shared.data_ptr<int32_t>(), exceptions.data_ptr<int32_t>(), optr, q.size(1), k.size(1), k.size(2), shared.size(2), exceptions.size(2));
  } else if (mode >= 5) {
    const dim3 grid(q.size(0), q.size(1));
    const auto stream = at::cuda::getCurrentCUDAStream();
    if (mode == 5)
      gas_token_parallel_kernel<4><<<grid, 128, 0, stream>>>(qptr, kptr, vptr, shared.data_ptr<int32_t>(), exceptions.data_ptr<int32_t>(), optr, q.size(1), k.size(1), k.size(2), shared.size(2), exceptions.size(2));
    else if (mode == 6)
      gas_token_parallel_kernel<8><<<grid, 256, 0, stream>>>(qptr, kptr, vptr, shared.data_ptr<int32_t>(), exceptions.data_ptr<int32_t>(), optr, q.size(1), k.size(1), k.size(2), shared.size(2), exceptions.size(2));
    else
      gas_token_parallel_kernel<16><<<grid, 512, 0, stream>>>(qptr, kptr, vptr, shared.data_ptr<int32_t>(), exceptions.data_ptr<int32_t>(), optr, q.size(1), k.size(1), k.size(2), shared.size(2), exceptions.size(2));
  } else if (mode == 2 || mode == 3) {
    TORCH_CHECK(tile > 0, "tile must be positive");
    const int chunks = (shared.size(2) + exceptions.size(2) + tile - 1) / tile;
    TORCH_CHECK(chunks > 0 && chunks <= 65535, "invalid chunk count");
    auto partial = torch::empty({q.size(0), q.size(1), chunks, 130}, q.options().dtype(at::kFloat));
    if (mode == 3) {
      TORCH_CHECK(tile * 128 * 2 * sizeof(half) <= 48 * 1024, "tile too large");
      gas_split_kernel<true><<<dim3(q.size(0), k.size(1), chunks), 128, tile * 128 * 2 * sizeof(half), at::cuda::getCurrentCUDAStream()>>>(
          qptr, kptr, vptr, shared.data_ptr<int32_t>(), exceptions.data_ptr<int32_t>(), partial.data_ptr<float>(),
          q.size(1), k.size(1), k.size(2), shared.size(2), exceptions.size(2), chunks, tile);
    } else {
      gas_split_kernel<false><<<dim3(q.size(0), k.size(1), chunks), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
          qptr, kptr, vptr, shared.data_ptr<int32_t>(), exceptions.data_ptr<int32_t>(), partial.data_ptr<float>(),
          q.size(1), k.size(1), k.size(2), shared.size(2), exceptions.size(2), chunks, tile);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    gas_merge_kernel<<<dim3(q.size(0), q.size(1)), 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        partial.data_ptr<float>(), optr, q.size(1), chunks);
  } else if (mode == 4) {
    gas_tiled_head_kernel<<<dim3(q.size(0), q.size(1)), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        qptr, kptr, vptr, shared.data_ptr<int32_t>(), exceptions.data_ptr<int32_t>(), optr,
        q.size(1), k.size(1), k.size(2), shared.size(2), exceptions.size(2));
  } else if (mode == 1) {
    TORCH_CHECK((shared.size(2) + exceptions.size(2)) * sizeof(float) <= 48 * 1024, "support too large");
    gas_head_parallel_kernel<<<dim3(q.size(0), q.size(1)), 128, (shared.size(2) + exceptions.size(2)) * sizeof(float), at::cuda::getCurrentCUDAStream()>>>(
        qptr, kptr, vptr, shared.data_ptr<int32_t>(), exceptions.data_ptr<int32_t>(), optr,
        q.size(1), k.size(1), k.size(2), shared.size(2), exceptions.size(2));
  } else {
    gas_warp_kernel<<<dim3(q.size(0), k.size(1)), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        qptr, kptr, vptr, shared.data_ptr<int32_t>(), exceptions.data_ptr<int32_t>(), optr,
        q.size(1), k.size(1), k.size(2), shared.size(2), exceptions.size(2));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode", &gas_decode, pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
        pybind11::arg("shared"), pybind11::arg("exceptions"), pybind11::arg("mode") = 0,
        pybind11::arg("tile") = 16);
}
