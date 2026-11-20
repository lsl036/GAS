#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>
#include <cstdint>

template<typename T> __device__ __forceinline__ float cvt(T x);
template<> __device__ __forceinline__ float cvt<half>(half x) { return __half2float(x); }
template<> __device__ __forceinline__ float cvt<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }
template<typename T> __device__ __forceinline__ T out_cvt(float x);
template<> __device__ __forceinline__ half out_cvt<half>(float x) { return __float2half(x); }
template<> __device__ __forceinline__ __nv_bfloat16 out_cvt<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

// Prefix counts, unique indices and disjoint exceptions are trusted metadata.
template<typename T, int Segments>
__device__ void group_attention_body(const T* __restrict__ q, const T* __restrict__ k,
                                const T* __restrict__ v, const int32_t* __restrict__ shared,
                                const int32_t* __restrict__ ex, const int32_t* __restrict__ sc,
                                const int32_t* __restrict__ ec, T* __restrict__ out, float* __restrict__ partial_out,
                                int heads, int groups, int length, int sw, int ew, int max_parts, int chunk,
                                int batch, int group, int part) {
  const int lane = threadIdx.x & 15, segment = threadIdx.x >> 4;
  const int64_t qbase = (static_cast<int64_t>(batch) * heads + group * 4) * 128;
  const int64_t kvbase = (static_cast<int64_t>(batch) * groups + group) * length * 128;
  const int64_t sbase = (static_cast<int64_t>(batch) * groups + group) * sw;
  const int64_t ebase = (static_cast<int64_t>(batch) * heads + group * 4) * ew;
  int max_count = sc[batch * groups + group];
  #pragma unroll
  for (int h = 0; h < 4; ++h) max_count = max(max_count, ec[batch * heads + group * 4 + h]);
  const int parts = min(max_parts, max(1, (max_count + chunk - 1) / chunk));
  if (part >= parts) return;
  float query[4][8], acc[4][8], maximum[4], sums[4];
  #pragma unroll
  for (int h = 0; h < 4; ++h) {
    const uint4 raw = *reinterpret_cast<const uint4*>(q + qbase + h * 128 + lane * 8);
    const T* values = reinterpret_cast<const T*>(&raw);
    #pragma unroll
    for (int i = 0; i < 8; ++i) { query[h][i] = cvt<T>(values[i]); acc[h][i] = 0; }
    maximum[h] = -INFINITY;
    sums[h] = 0;
  }
  const int shared_count = sc[batch * groups + group];
  const int shared_span = (shared_count + parts - 1) / parts;
  const int shared_start = part * shared_span, shared_end = min(shared_count, shared_start + shared_span);
  for (int pos = shared_start + segment; pos < shared_end; pos += Segments) {
    const int token = shared[sbase + pos];
    if (token < 0 || token >= length) continue;
    const int64_t offset = kvbase + static_cast<int64_t>(token) * 128 + lane * 8;
    const uint4 kr = *reinterpret_cast<const uint4*>(k + offset);
    const uint4 vr = *reinterpret_cast<const uint4*>(v + offset);
    const T* kval = reinterpret_cast<const T*>(&kr);
    const T* vval = reinterpret_cast<const T*>(&vr);
    #pragma unroll
    for (int h = 0; h < 4; ++h) {
      float dot = 0;
      #pragma unroll
      for (int i = 0; i < 8; ++i) dot += query[h][i] * cvt<T>(kval[i]);
      const unsigned active = __activemask();
      for (int d = 8; d; d >>= 1) dot += __shfl_down_sync(active, dot, d, 16);
      dot = __shfl_sync(active, dot, 0, 16) * 0.08838834764831845f;
      const float next = fmaxf(maximum[h], dot), a = __expf(maximum[h] - next), b = __expf(dot - next);
      #pragma unroll
      for (int i = 0; i < 8; ++i) acc[h][i] = acc[h][i] * a + b * cvt<T>(vval[i]);
      sums[h] = sums[h] * a + b;
      maximum[h] = next;
    }
  }
  #pragma unroll
  for (int h = 0; h < 4; ++h) {
    const int count = ec[batch * heads + group * 4 + h];
    const int span = (count + parts - 1) / parts;
    const int start = part * span, end = min(count, start + span);
    for (int pos = start + segment; pos < end; pos += Segments) {
      const int token = ex[ebase + h * ew + pos];
      if (token < 0 || token >= length) continue;
      const int64_t offset = kvbase + static_cast<int64_t>(token) * 128 + lane * 8;
      const uint4 kr = *reinterpret_cast<const uint4*>(k + offset);
      const uint4 vr = *reinterpret_cast<const uint4*>(v + offset);
      const T* kval = reinterpret_cast<const T*>(&kr);
      const T* vval = reinterpret_cast<const T*>(&vr);
      float dot = 0;
      #pragma unroll
      for (int i = 0; i < 8; ++i) dot += query[h][i] * cvt<T>(kval[i]);
      const unsigned active = __activemask();
      for (int d = 8; d; d >>= 1) dot += __shfl_down_sync(active, dot, d, 16);
      dot = __shfl_sync(active, dot, 0, 16) * 0.08838834764831845f;
      const float next = fmaxf(maximum[h], dot), a = __expf(maximum[h] - next), b = __expf(dot - next);
      #pragma unroll
      for (int i = 0; i < 8; ++i) acc[h][i] = acc[h][i] * a + b * cvt<T>(vval[i]);
      sums[h] = sums[h] * a + b;
      maximum[h] = next;
    }
  }
  __shared__ float partial[Segments][4][8][16], meta[Segments][4][2];
  #pragma unroll
  for (int h = 0; h < 4; ++h) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) partial[segment][h][i][lane] = acc[h][i];
    if (!lane) { meta[segment][h][0] = maximum[h]; meta[segment][h][1] = sums[h]; }
  }
  __syncthreads();
  if (segment == 0) {
    #pragma unroll
    for (int h = 0; h < 4; ++h) {
      float maxval = -INFINITY, denominator = 0, result[8] = {0};
      #pragma unroll
      for (int s = 0; s < Segments; ++s) maxval = fmaxf(maxval, meta[s][h][0]);
      #pragma unroll
      for (int s = 0; s < Segments; ++s) {
        if (meta[s][h][1] == 0) continue;
        const float weight = __expf(meta[s][h][0] - maxval);
        denominator += meta[s][h][1] * weight;
        #pragma unroll
        for (int i = 0; i < 8; ++i) result[i] += partial[s][h][i][lane] * weight;
      }
      if (parts == 1) {
        uint4 raw;
        T* values = reinterpret_cast<T*>(&raw);
        #pragma unroll
        for (int i = 0; i < 8; ++i) values[i] = out_cvt<T>(result[i] / denominator);
        *reinterpret_cast<uint4*>(out + qbase + h * 128 + lane * 8) = raw;
      } else {
        const int64_t base = (((static_cast<int64_t>(batch) * groups + group) * max_parts + part) * 4 + h) * 130;
        #pragma unroll
        for (int i = 0; i < 8; ++i) partial_out[base + lane * 8 + i] = result[i];
        if (!lane) { partial_out[base + 128] = maxval; partial_out[base + 129] = denominator; }
      }
    }
  }
}

template<typename T, int Segments>
__global__ void group_attention(const T* q, const T* k, const T* v, const int32_t* shared,
                                const int32_t* ex, const int32_t* sc, const int32_t* ec,
                                T* out, float* partial, int heads, int groups, int length,
                                int sw, int ew, int parts, int chunk) {
  group_attention_body<T, Segments>(q, k, v, shared, ex, sc, ec, out, partial, heads, groups,
                                   length, sw, ew, parts, chunk, blockIdx.x, blockIdx.y, blockIdx.z);
}

// One block builds offsets for at most 1024 KV groups and resets the queue.
__global__ void compact_group_plan(const int32_t* sc, const int32_t* ec, int32_t* offsets,
                                   int32_t* counter, int total_groups, int max_parts, int chunk) {
  using Scan = cub::BlockScan<int, 256>;
  __shared__ typename Scan::TempStorage temp;
  int counts[4], starts[4], total;
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int group = threadIdx.x * 4 + i;
    int count = 0;
    if (group < total_groups) {
      int maximum = sc[group];
      #pragma unroll
      for (int h = 0; h < 4; ++h) maximum = max(maximum, ec[group * 4 + h]);
      count = min(max_parts, max(1, (maximum + chunk - 1) / chunk));
    }
    counts[i] = count;
  }
  Scan(temp).ExclusiveSum(counts, starts, total);
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int group = threadIdx.x * 4 + i;
    if (group < total_groups) offsets[group] = starts[i];
  }
  if (threadIdx.x == 0) { offsets[total_groups] = total; *counter = 0; }
}

template<typename T, int Segments>
__global__ void compact_group_attention(const T* q, const T* k, const T* v, const int32_t* shared,
                                        const int32_t* ex, const int32_t* sc, const int32_t* ec,
                                        T* out, float* partial, const int32_t* offsets, int32_t* counter,
                                        int heads, int groups, int length, int sw, int ew,
                                        int parts, int chunk, int total_groups) {
  __shared__ int task, selected_group, selected_part;
  while (true) {
    if (threadIdx.x == 0) task = atomicAdd(counter, 1);
    __syncthreads();
    if (task >= offsets[total_groups]) return;
    if (threadIdx.x == 0) {
      int lo = 0, hi = total_groups;
      while (lo + 1 < hi) {
        const int mid = (lo + hi) / 2;
        if (offsets[mid] <= task) lo = mid; else hi = mid;
      }
      selected_group = lo;
      selected_part = task - offsets[lo];
    }
    __syncthreads();
    group_attention_body<T, Segments>(q, k, v, shared, ex, sc, ec, out, partial, heads, groups,
                                     length, sw, ew, parts, chunk, selected_group / groups,
                                     selected_group % groups, selected_part);
    __syncthreads();
  }
}

template<typename T>
__global__ void merge_parts(const float* __restrict__ partial, T* __restrict__ out,
                            const int32_t* __restrict__ sc, const int32_t* __restrict__ ec,
                            int heads, int groups, int max_parts, int chunk) {
  const int batch = blockIdx.x, head = blockIdx.y, group = head / 4, lane = threadIdx.x;
  int max_count = sc[batch * groups + group];
  #pragma unroll
  for (int h = 0; h < 4; ++h) max_count = max(max_count, ec[batch * heads + group * 4 + h]);
  const int parts = min(max_parts, max(1, (max_count + chunk - 1) / chunk));
  if (parts == 1) return;
  const int64_t group_base = (static_cast<int64_t>(batch) * groups + group) * max_parts * 4 * 130;
  const int local_head = head % 4;
  float maxval = -INFINITY, denominator = 0, result[8] = {0};
  for (int p = 0; p < parts; ++p) maxval = fmaxf(maxval, partial[group_base + (p * 4 + local_head) * 130 + 128]);
  for (int p = 0; p < parts; ++p) {
    const int64_t base = group_base + (p * 4 + local_head) * 130;
    const float sum = partial[base + 129];
    if (sum == 0) continue;
    const float weight = __expf(partial[base + 128] - maxval);
    denominator += sum * weight;
    #pragma unroll
    for (int i = 0; i < 8; ++i) result[i] += partial[base + lane * 8 + i] * weight;
  }
  uint4 raw;
  T* values = reinterpret_cast<T*>(&raw);
  #pragma unroll
  for (int i = 0; i < 8; ++i) values[i] = out_cvt<T>(result[i] / denominator);
  *reinterpret_cast<uint4*>(out + (static_cast<int64_t>(batch) * heads + head) * 128 + lane * 8) = raw;
}

template<typename T> void launch(torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor shared, torch::Tensor ex, torch::Tensor sc, torch::Tensor ec, torch::Tensor out, int segments, int parts, int chunk) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  auto partial = torch::empty({q.size(0), k.size(1), parts, 4, 130}, q.options().dtype(at::kFloat));
  const dim3 grid(q.size(0), k.size(1), parts);
  #define RUN(N) group_attention<T,N><<<grid, N * 16, 0, stream>>>(reinterpret_cast<const T*>(q.data_ptr()), reinterpret_cast<const T*>(k.data_ptr()), reinterpret_cast<const T*>(v.data_ptr()), shared.data_ptr<int32_t>(), ex.data_ptr<int32_t>(), sc.data_ptr<int32_t>(), ec.data_ptr<int32_t>(), reinterpret_cast<T*>(out.data_ptr()), partial.data_ptr<float>(), q.size(1), k.size(1), k.size(2), shared.size(2), ex.size(2), parts, chunk)
  if (segments == 4) { RUN(4); } else if (segments == 8) { RUN(8); } else { RUN(16); }
  #undef RUN
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  merge_parts<T><<<dim3(q.size(0), q.size(1)), 16, 0, stream>>>(partial.data_ptr<float>(), reinterpret_cast<T*>(out.data_ptr()), sc.data_ptr<int32_t>(), ec.data_ptr<int32_t>(), q.size(1), k.size(1), parts, chunk);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename T> void launch_compact(torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor shared, torch::Tensor ex, torch::Tensor sc, torch::Tensor ec, torch::Tensor out,
    int segments, int parts, int chunk, int blocks) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int total_groups = q.size(0) * k.size(1);
  auto partial = torch::empty({q.size(0), k.size(1), parts, 4, 130}, q.options().dtype(at::kFloat));
  auto offsets = torch::empty({total_groups + 1}, shared.options());
  auto counter = torch::empty({1}, shared.options());
  compact_group_plan<<<1, 256, 0, stream>>>(sc.data_ptr<int32_t>(), ec.data_ptr<int32_t>(),
      offsets.data_ptr<int32_t>(), counter.data_ptr<int32_t>(), total_groups, parts, chunk);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  #define RUN_COMPACT(N) compact_group_attention<T,N><<<blocks, N * 16, 0, stream>>>(reinterpret_cast<const T*>(q.data_ptr()), reinterpret_cast<const T*>(k.data_ptr()), reinterpret_cast<const T*>(v.data_ptr()), shared.data_ptr<int32_t>(), ex.data_ptr<int32_t>(), sc.data_ptr<int32_t>(), ec.data_ptr<int32_t>(), reinterpret_cast<T*>(out.data_ptr()), partial.data_ptr<float>(), offsets.data_ptr<int32_t>(), counter.data_ptr<int32_t>(), q.size(1), k.size(1), k.size(2), shared.size(2), ex.size(2), parts, chunk, total_groups)
  if (segments == 4) { RUN_COMPACT(4); } else if (segments == 8) { RUN_COMPACT(8); } else { RUN_COMPACT(16); }
  #undef RUN_COMPACT
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  merge_parts<T><<<dim3(q.size(0), q.size(1)), 16, 0, stream>>>(partial.data_ptr<float>(), reinterpret_cast<T*>(out.data_ptr()), sc.data_ptr<int32_t>(), ec.data_ptr<int32_t>(), q.size(1), k.size(1), parts, chunk);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor decode_impl(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor shared,
                          torch::Tensor ex, torch::Tensor sc, torch::Tensor ec, int segments, int parts, int chunk, int blocks) {
  for (const auto& x : {q,k,v,shared,ex,sc,ec}) TORCH_CHECK(x.is_cuda() && x.device() == q.device() && x.is_contiguous(), "contiguous tensors on one CUDA device required");
  TORCH_CHECK((q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16) && q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(), "FP16 or BF16 required");
  for (const auto& x : {shared,ex,sc,ec}) TORCH_CHECK(x.scalar_type() == at::kInt, "int32 metadata required");
  TORCH_CHECK(q.dim() == 3 && k.dim() == 4 && k.sizes() == v.sizes() && q.size(0) == k.size(0) && q.size(1) == 4*k.size(1) && q.size(2) == 128 && k.size(3) == 128, "invalid Q/K/V shape");
  TORCH_CHECK(shared.dim() == 3 && shared.size(0) == q.size(0) && shared.size(1) == k.size(1) && ex.dim() == 3 && ex.size(0) == q.size(0) && ex.size(1) == q.size(1), "invalid support shape");
  TORCH_CHECK(sc.dim() == 2 && sc.size(0) == q.size(0) && sc.size(1) == k.size(1) && ec.dim() == 2 && ec.size(0) == q.size(0) && ec.size(1) == q.size(1), "invalid count shape");
  TORCH_CHECK(k.size(2) > 0 && shared.size(2) + ex.size(2) > 0 && (segments == 4 || segments == 8 || segments == 16), "invalid size or segments");
  TORCH_CHECK((parts == 2 || parts == 4 || parts == 8 || parts == 16 || parts == 32) && chunk > 0, "invalid partition configuration");
  TORCH_CHECK(blocks >= 0 && blocks <= 4096 && (blocks == 0 || q.size(0) * k.size(1) <= 1024), "invalid compact work grid");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(k.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(v.data_ptr()) % 16 == 0, "16-byte alignment required");
  const c10::cuda::CUDAGuard guard(q.device());
  auto out = torch::empty_like(q);
  if (q.scalar_type() == at::kHalf) {
    if (blocks) launch_compact<half>(q,k,v,shared,ex,sc,ec,out,segments,parts,chunk,blocks);
    else launch<half>(q,k,v,shared,ex,sc,ec,out,segments,parts,chunk);
  } else {
    if (blocks) launch_compact<__nv_bfloat16>(q,k,v,shared,ex,sc,ec,out,segments,parts,chunk,blocks);
    else launch<__nv_bfloat16>(q,k,v,shared,ex,sc,ec,out,segments,parts,chunk);
  }
  return out;
}

torch::Tensor decode(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor shared,
                     torch::Tensor ex, torch::Tensor sc, torch::Tensor ec, int segments, int parts, int chunk) {
  return decode_impl(q,k,v,shared,ex,sc,ec,segments,parts,chunk,0);
}

torch::Tensor decode_compact(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor shared,
                             torch::Tensor ex, torch::Tensor sc, torch::Tensor ec, int segments, int parts, int chunk, int blocks) {
  TORCH_CHECK(blocks > 0, "compact blocks must be positive");
  return decode_impl(q,k,v,shared,ex,sc,ec,segments,parts,chunk,blocks);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("decode", &decode); m.def("decode_compact", &decode_compact); }
