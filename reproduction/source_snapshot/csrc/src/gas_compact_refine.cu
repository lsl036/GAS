#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <torch/extension.h>

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cstdint>
#include <vector>

namespace {
constexpr int THREADS = 256;
using Reduce = cub::BlockReduce<float, THREADS>;

struct PivotStats { float sum, min_gt, max_le; };
struct CombinePivot {
  __device__ PivotStats operator()(PivotStats a, PivotStats b) const {
    return {a.sum + b.sum, fminf(a.min_gt, b.min_gt), fmaxf(a.max_le, b.max_le)};
  }
};
using PivotReduce = cub::BlockReduce<PivotStats, THREADS>;

struct MassStats { float total, sink; };
struct CombineMass {
  __device__ MassStats operator()(MassStats a, MassStats b) const {
    return {a.total + b.total, a.sink + b.sink};
  }
};
using MassReduce = cub::BlockReduce<MassStats, THREADS>;

template <bool HEAD_TAGGED>
__device__ bool head_selected(const bool* mask, const int32_t* head_epochs, int head, int64_t offset) {
  return HEAD_TAGGED ? reinterpret_cast<const int32_t*>(mask)[offset] == head_epochs[head] : mask[offset];
}

template <typename T, bool TAGGED, bool SHARED, bool DENSE, bool FUSED_REDUCE, bool WARP_COMPACT,
          bool FUSED_EXCEPTION = false, bool FAST_GROUP_SCORES = false, bool FUSED_MASS = false,
          bool HEAD_TAGGED = false, bool NORMALIZE_ONCE = false, bool PREPARED = false>
__global__ void group_refine(const int32_t* __restrict__ ids,
                             const int32_t* __restrict__ counts,
                             const bool* __restrict__ mask,
                             const T* __restrict__ scores,
                             int32_t* __restrict__ group_ids,
                             int32_t* __restrict__ group_counts,
                             uint8_t* __restrict__ flags,
                             int32_t* __restrict__ epochs,
                             float* __restrict__ scratch,
                             int length, int cap, int shared_cap, float threshold,
                             int32_t* __restrict__ fused_ex_ids = nullptr,
                             int32_t* __restrict__ fused_ex_counts = nullptr,
                             const int32_t* __restrict__ head_epochs = nullptr) {
  const int group_row = blockIdx.x;
  const int batch = group_row / 8, group = group_row % 8, tid = threadIdx.x;
  const int head_base = batch * 32 + group * 4;
  const int out_cap = 4 * cap;
  __shared__ typename Reduce::TempStorage temp;
  __shared__ typename PivotReduce::TempStorage pivot_temp;
  __shared__ typename MassReduce::TempStorage mass_temp;
  extern __shared__ float local_scratch[];
  __shared__ float maximum, total, sink, low, high, min_gt, max_le, mass_above;
  __shared__ int32_t epoch;
  __shared__ float warp_scores[8];
  __shared__ int warp_tokens[8], warp_counts[8], selected_tokens[4], max_ex_count;
  if (tid == 0) {
    group_counts[group_row] = 0;
    if (FUSED_EXCEPTION) {
      for (int h = 0; h < 4; ++h) fused_ex_counts[head_base + h] = 0;
    }
    if (TAGGED) { epoch = epochs[group_row] + 1; epochs[group_row] = epoch; }
  }
  __syncthreads();
  const int n0 = counts[head_base], n1 = counts[head_base + 1];
  const int n2 = counts[head_base + 2], n3 = counts[head_base + 3];
  const int total_candidates = n0 + n1 + n2 + n3;
  int work_capacity = DENSE ? length : 4 * cap;
  float* work = SHARED && total_candidates <= shared_cap
                    ? local_scratch : scratch + static_cast<int64_t>(group_row) * work_capacity;
  float local_max = -FLT_MAX;
  int head_offset = 0;
  for (int h = 0; h < (DENSE ? 1 : 4); ++h) {
    int head = head_base + h;
    int size = DENSE ? length : counts[head];
    for (int pos = tid; pos < size; pos += THREADS) {
      if (PREPARED) {
        local_max = fmaxf(local_max, work[head_offset + pos]);
        continue;
      }
      int token = DENSE ? pos : ids[static_cast<int64_t>(head) * cap + pos];
      bool duplicate = false;
      for (int prev = 0; !DENSE && prev < h; ++prev)
        duplicate |= head_selected<HEAD_TAGGED>(mask, head_epochs, head_base + prev,
                                                 static_cast<int64_t>(head_base + prev) * length + token);
      float value = -FLT_MAX;
      if (!duplicate) {
        if (FAST_GROUP_SCORES) {
          // A unique compact candidate belongs to its own head and cannot
          // belong to an earlier head. Only later heads need checking.
          value = static_cast<float>(scores[static_cast<int64_t>(head) * length + token]);
          for (int hh = h + 1; hh < 4; ++hh) {
            int64_t offset = static_cast<int64_t>(head_base + hh) * length + token;
            if (head_selected<HEAD_TAGGED>(mask, head_epochs, head_base + hh, offset))
              value = fmaxf(value, static_cast<float>(scores[offset]));
          }
        } else {
          for (int hh = 0; hh < 4; ++hh) {
            int64_t offset = static_cast<int64_t>(head_base + hh) * length + token;
            if (head_selected<HEAD_TAGGED>(mask, head_epochs, head_base + hh, offset))
              value = fmaxf(value, static_cast<float>(scores[offset]));
          }
        }
      }
      work[head_offset + pos] = value;
      local_max = fmaxf(local_max, value);
    }
    head_offset += size;
  }
  float reduced = Reduce(temp).Reduce(local_max, cub::Max());
  if (tid == 0) maximum = reduced;
  __syncthreads();
  float local_sum = 0.0f, local_sink = 0.0f;
  head_offset = 0;
  for (int h = 0; h < (DENSE ? 1 : 4); ++h) {
    int head = head_base + h;
    int size = DENSE ? length : counts[head];
    for (int pos = tid; pos < size; pos += THREADS) {
      int idx = head_offset + pos;
      float val = work[idx];
      float weight = val == -FLT_MAX ? 0.0f : expf(val - maximum);
      work[idx] = weight;
      local_sum += weight;
      int token = DENSE ? pos : ids[static_cast<int64_t>(head) * cap + pos];
      if (token == 0) local_sink += weight;
    }
    head_offset += size;
  }
  if (FUSED_MASS) {
    MassStats stats = MassReduce(mass_temp).Reduce(MassStats{local_sum, local_sink}, CombineMass{});
    if (tid == 0) {
      total = stats.total; sink = stats.sink; low = 0.0f;
      high = total > 0 ? sink / total : 0.0f;
    }
  } else {
    reduced = Reduce(temp).Sum(local_sum);
    if (tid == 0) total = reduced;
    __syncthreads();
    reduced = Reduce(temp).Sum(local_sink);
    if (tid == 0) { sink = reduced; low = 0.0f; high = total > 0 ? sink / total : 0.0f; }
  }
  __syncthreads();
  if (total <= 0.0f) return;
  if (NORMALIZE_ONCE) {
    for (int pos = tid; pos < total_candidates; pos += THREADS) work[pos] /= total;
    __syncthreads();
  }
  bool again;
  do {
    float mid = (low + high) * 0.5f;
    float lsum = 0.0f, lmin = high, lmax = low;
    head_offset = 0;
    for (int h = 0; h < (DENSE ? 1 : 4); ++h) {
      int head = head_base + h;
      int size = DENSE ? length : counts[head];
      for (int pos = tid; pos < size; pos += THREADS) {
        float prob = NORMALIZE_ONCE ? work[head_offset + pos] : work[head_offset + pos] / total;
        if (prob > mid) lsum += prob;
        if (prob > low) lmin = fminf(lmin, prob);
        if (prob <= high) lmax = fmaxf(lmax, prob);
      }
      head_offset += size;
    }
    if (FUSED_REDUCE) {
      PivotStats values = PivotReduce(pivot_temp).Reduce(PivotStats{lsum, lmin, lmax}, CombinePivot{});
      if (tid == 0) {
        mass_above = values.sum; min_gt = values.min_gt; max_le = values.max_le;
        if (mass_above >= threshold) low = mid;
        else high = fminf(mid, max_le);
      }
    } else {
      reduced = Reduce(temp).Sum(lsum);
      if (tid == 0) mass_above = reduced;
      __syncthreads();
      reduced = Reduce(temp).Reduce(lmin, cub::Min());
      if (tid == 0) min_gt = reduced;
      __syncthreads();
      reduced = Reduce(temp).Reduce(lmax, cub::Max());
      if (tid == 0) {
        max_le = reduced;
        if (mass_above >= threshold) low = mid;
        else high = fminf(mid, max_le);
      }
    }
    __syncthreads();
    again = max_le - min_gt > 1.0e-5f;
  } while (again);
  head_offset = 0;
  for (int h = 0; h < (DENSE ? 1 : 4); ++h) {
    int head = head_base + h;
    int size = DENSE ? length : counts[head];
    for (int pos = tid; pos < size; pos += THREADS) {
      float prob = NORMALIZE_ONCE ? work[head_offset + pos] : work[head_offset + pos] / total;
      if (prob > low) {
        int token = DENSE ? pos : ids[static_cast<int64_t>(head) * cap + pos];
        int rank;
        if (WARP_COMPACT) {
          unsigned active = __activemask();
          int lane = tid & 31;
          int leader = __ffs(active) - 1;
          int base = 0;
          if (lane == leader) base = atomicAdd(group_counts + group_row, __popc(active));
          base = __shfl_sync(active, base, leader);
          rank = base + __popc(active & ((1u << lane) - 1u));
        } else {
          rank = atomicAdd(group_counts + group_row, 1);
        }
        group_ids[static_cast<int64_t>(group_row) * out_cap + rank] = token;
        int64_t flag_idx = static_cast<int64_t>(group_row) * length + token;
        if (TAGGED) reinterpret_cast<int32_t*>(flags)[flag_idx] = epoch;
        else flags[flag_idx] = 1;
      }
    }
    head_offset += size;
  }
  if (FUSED_EXCEPTION) {
    // Each head is handled by two warps. Keep only the best 16 entries per
    // thread; a parallel merge then produces the exact group-excluded top 16.
    __syncthreads();
    const int head = head_base + tid / 64;
    const int lane64 = tid % 64, lane = tid % 32, warp = tid / 32;
    float best_scores[16];
    int best_tokens[16];
    #pragma unroll
    for (int j = 0; j < 16; ++j) { best_scores[j] = -FLT_MAX; best_tokens[j] = INT_MAX; }
    int residual_count = 0;
    for (int pos = lane64; pos < counts[head]; pos += 64) {
      int token = ids[static_cast<int64_t>(head) * cap + pos];
      if (reinterpret_cast<int32_t*>(flags)[static_cast<int64_t>(group_row) * length + token] == epoch) continue;
      ++residual_count;
      float value = static_cast<float>(scores[static_cast<int64_t>(head) * length + token]);
      #pragma unroll
      for (int j = 0; j < 16; ++j) {
        bool better = value > best_scores[j] || (value == best_scores[j] && token < best_tokens[j]);
        float saved_value = best_scores[j];
        int saved_token = best_tokens[j];
        if (better) { best_scores[j] = value; best_tokens[j] = token; value = saved_value; token = saved_token; }
      }
    }
    for (int offset = 16; offset; offset >>= 1) residual_count += __shfl_down_sync(0xffffffff, residual_count, offset);
    if (lane == 0) warp_counts[warp] = residual_count;
    __syncthreads();
    if (lane64 == 0) fused_ex_counts[head] = min(16, warp_counts[warp] + warp_counts[warp + 1]);
    __syncthreads();
    if (tid == 0) max_ex_count = max(max(fused_ex_counts[head_base], fused_ex_counts[head_base + 1]),
                                      max(fused_ex_counts[head_base + 2], fused_ex_counts[head_base + 3]));
    __syncthreads();
    int cursor = 0;
    for (int rank = 0; rank < max_ex_count; ++rank) {
      float value = cursor < 16 ? best_scores[cursor] : -FLT_MAX;
      int token = cursor < 16 ? best_tokens[cursor] : INT_MAX;
      for (int offset = 16; offset; offset >>= 1) {
        float other_value = __shfl_down_sync(0xffffffff, value, offset);
        int other_token = __shfl_down_sync(0xffffffff, token, offset);
        if (other_value > value || (other_value == value && other_token < token)) {
          value = other_value; token = other_token;
        }
      }
      if (lane == 0) { warp_scores[warp] = value; warp_tokens[warp] = token; }
      __syncthreads();
      if (lane64 == 0) {
        float other_value = warp_scores[warp + 1];
        int other_token = warp_tokens[warp + 1];
        int picked = (other_value > value || (other_value == value && other_token < token)) ? other_token : token;
        selected_tokens[tid / 64] = picked;
        if (rank < fused_ex_counts[head]) fused_ex_ids[static_cast<int64_t>(head) * 16 + rank] = picked;
      }
      __syncthreads();
      if (cursor < 16 && best_tokens[cursor] == selected_tokens[tid / 64]) ++cursor;
    }
  }
}

template <typename T>
__global__ void prepare_group_scores(const int32_t* __restrict__ ids,
                                     const int32_t* __restrict__ counts,
                                     const bool* __restrict__ mask,
                                     const T* __restrict__ scores,
                                     float* __restrict__ scratch, int length, int cap) {
  int head = blockIdx.x, group = head / 4, within_group = head % 4;
  int base = group * 4;
  int offset = 0;
  for (int h = 0; h < within_group; ++h) offset += counts[base + h];
  float* work = scratch + static_cast<int64_t>(group) * 4 * cap;
  for (int pos = threadIdx.x; pos < counts[head]; pos += blockDim.x) {
    int token = ids[static_cast<int64_t>(head) * cap + pos];
    bool duplicate = false;
    for (int h = 0; h < within_group; ++h)
      duplicate |= mask[static_cast<int64_t>(base + h) * length + token];
    float value = -FLT_MAX;
    if (!duplicate) {
      for (int h = 0; h < 4; ++h) {
        int64_t index = static_cast<int64_t>(base + h) * length + token;
        if (mask[index]) value = fmaxf(value, static_cast<float>(scores[index]));
      }
    }
    work[offset + pos] = value;
  }
}

struct ScorePos { float score; int pos; };
struct MaxScorePos {
  __device__ ScorePos operator()(ScorePos a, ScorePos b) const {
    return (a.score > b.score || (a.score == b.score && a.pos < b.pos)) ? a : b;
  }
};
using PairReduce = cub::BlockReduce<ScorePos, THREADS>;

template <bool TAGGED>
__device__ bool is_selected(const uint8_t* flags, const int32_t* epochs, int group_row, int length, int token) {
  int64_t idx = static_cast<int64_t>(group_row) * length + token;
  return TAGGED ? reinterpret_cast<const int32_t*>(flags)[idx] == epochs[group_row] : flags[idx] != 0;
}

template <typename T, bool TAGGED>
__global__ void exceptions(const int32_t* __restrict__ ids,
                           const int32_t* __restrict__ counts,
                           const T* __restrict__ scores,
                           const uint8_t* __restrict__ flags,
                           const int32_t* __restrict__ epochs,
                           int32_t* __restrict__ ex_ids,
                           int32_t* __restrict__ ex_counts,
                           int length, int cap, int budget) {
  int head = blockIdx.x, batch = head / 32, group = (head % 32) / 4, tid = threadIdx.x;
  __shared__ typename cub::BlockReduce<int, THREADS>::TempStorage sum_temp;
  __shared__ typename PairReduce::TempStorage pair_temp;
  __shared__ int residual_count;
  int local = 0;
  for (int pos = tid; pos < counts[head]; pos += THREADS) {
    int token = ids[static_cast<int64_t>(head) * cap + pos];
    local += !is_selected<TAGGED>(flags, epochs, batch * 8 + group, length, token);
  }
  int sum = cub::BlockReduce<int, THREADS>(sum_temp).Sum(local);
  if (tid == 0) { residual_count = sum; ex_counts[head] = 0; }
  __syncthreads();
  if (residual_count <= budget) {
    for (int pos = tid; pos < counts[head]; pos += THREADS) {
      int token = ids[static_cast<int64_t>(head) * cap + pos];
      if (!is_selected<TAGGED>(flags, epochs, batch * 8 + group, length, token)) {
        int rank = atomicAdd(ex_counts + head, 1);
        ex_ids[static_cast<int64_t>(head) * budget + rank] = token;
      }
    }
    return;
  }
  for (int rank = 0; rank < budget; ++rank) {
    ScorePos best{-FLT_MAX, INT_MAX};
    for (int pos = tid; pos < counts[head]; pos += THREADS) {
      int token = ids[static_cast<int64_t>(head) * cap + pos];
      if (is_selected<TAGGED>(flags, epochs, batch * 8 + group, length, token)) continue;
      bool selected = false;
      for (int previous = 0; previous < rank; ++previous)
        selected |= ex_ids[static_cast<int64_t>(head) * budget + previous] == token;
      if (selected) continue;
      float score = static_cast<float>(scores[static_cast<int64_t>(head) * length + token]);
      best = MaxScorePos{}(best, ScorePos{score, pos});
    }
    ScorePos reduced = PairReduce(pair_temp).Reduce(best, MaxScorePos{});
    if (tid == 0) {
      ex_ids[static_cast<int64_t>(head) * budget + rank] =
          ids[static_cast<int64_t>(head) * cap + reduced.pos];
      ex_counts[head] = rank + 1;
    }
    __syncthreads();
  }
}

constexpr int TILE_ITEMS = 4;
constexpr int TILE_SIZE = THREADS * TILE_ITEMS;
constexpr int TILE_KEEP = 16;
using TileSort = cub::BlockRadixSort<float, THREADS, TILE_ITEMS, int32_t>;

template <typename T, bool TAGGED>
__global__ void exception_tiles(const int32_t* __restrict__ ids,
                                const int32_t* __restrict__ counts,
                                const T* __restrict__ scores,
                                const uint8_t* __restrict__ flags,
                                const int32_t* __restrict__ epochs,
                                float* __restrict__ tile_scores,
                                int32_t* __restrict__ tile_ids,
                                int length, int cap, int tiles) {
  const int head = blockIdx.x, tile = blockIdx.y, tid = threadIdx.x;
  const int batch = head / 32, group = (head % 32) / 4;
  __shared__ typename TileSort::TempStorage storage;
  if (counts[head] - tile * TILE_SIZE <= TILE_KEEP) {
    if (tid < TILE_KEEP) {
      int pos = tile * TILE_SIZE + tid;
      int token = pos < counts[head] ? ids[static_cast<int64_t>(head) * cap + pos] : -1;
      bool valid = token >= 0 && !is_selected<TAGGED>(flags, epochs, batch * 8 + group, length, token);
      int64_t out = (static_cast<int64_t>(head) * tiles + tile) * TILE_KEEP + tid;
      tile_scores[out] = valid ? static_cast<float>(scores[static_cast<int64_t>(head) * length + token]) : -FLT_MAX;
      tile_ids[out] = valid ? token : -1;
    }
    return;
  }
  float keys[TILE_ITEMS];
  int32_t values[TILE_ITEMS];
  #pragma unroll
  for (int j = 0; j < TILE_ITEMS; ++j) {
    int pos = tile * TILE_SIZE + tid * TILE_ITEMS + j;
    int token = pos < counts[head] ? ids[static_cast<int64_t>(head) * cap + pos] : -1;
    bool valid = token >= 0 && !is_selected<TAGGED>(flags, epochs, batch * 8 + group, length, token);
    keys[j] = valid ? static_cast<float>(scores[static_cast<int64_t>(head) * length + token]) : -FLT_MAX;
    values[j] = valid ? token : -1;
  }
  TileSort(storage).SortDescending(keys, values);
  #pragma unroll
  for (int j = 0; j < TILE_ITEMS; ++j) {
    int rank = tid * TILE_ITEMS + j;
    if (rank < TILE_KEEP) {
      int64_t out = (static_cast<int64_t>(head) * tiles + tile) * TILE_KEEP + rank;
      tile_scores[out] = keys[j];
      tile_ids[out] = values[j];
    }
  }
}

using MergeSort = cub::BlockRadixSort<float, THREADS, 1, int32_t>;
__global__ void exception_merge(const float* __restrict__ tile_scores,
                                const int32_t* __restrict__ tile_ids,
                                int32_t* __restrict__ ex_ids,
                                int32_t* __restrict__ ex_counts,
                                int tiles, int budget) {
  int head = blockIdx.x, tid = threadIdx.x;
  __shared__ typename MergeSort::TempStorage storage;
  __shared__ typename cub::BlockReduce<int, THREADS>::TempStorage reduce;
  float key[1] = {tid < tiles * TILE_KEEP ? tile_scores[(static_cast<int64_t>(head) * tiles * TILE_KEEP) + tid] : -FLT_MAX};
  int32_t val[1] = {tid < tiles * TILE_KEEP ? tile_ids[(static_cast<int64_t>(head) * tiles * TILE_KEEP) + tid] : -1};
  MergeSort(storage).SortDescending(key, val);
  if (tid < budget && val[0] >= 0) ex_ids[static_cast<int64_t>(head) * budget + tid] = val[0];
  int count = cub::BlockReduce<int, THREADS>(reduce).Sum(tid < budget && val[0] >= 0 ? 1 : 0);
  if (tid == 0) ex_counts[head] = count;
}

template <typename T, int ITEMS, bool TAGGED>
__global__ void exception_blocksort(const int32_t* __restrict__ ids,
                                    const int32_t* __restrict__ counts,
                                    const T* __restrict__ scores,
                                    const uint8_t* __restrict__ flags,
                                    const int32_t* __restrict__ epochs,
                                    int32_t* __restrict__ ex_ids,
                                    int32_t* __restrict__ ex_counts,
                                    int length, int cap, int budget, int low) {
  const int head = blockIdx.x, batch = head / 32, group = (head % 32) / 4, tid = threadIdx.x;
  const int count = counts[head];
  if (count <= low || count > THREADS * ITEMS) return;
  if (count <= budget) {
    if (tid == 0) ex_counts[head] = 0;
    __syncthreads();
    for (int pos = tid; pos < count; pos += THREADS) {
      int token = ids[static_cast<int64_t>(head) * cap + pos];
      if (!is_selected<TAGGED>(flags, epochs, batch * 8 + group, length, token)) {
        int rank = atomicAdd(ex_counts + head, 1);
        ex_ids[static_cast<int64_t>(head) * budget + rank] = token;
      }
    }
    return;
  }
  using Sort = cub::BlockRadixSort<float, THREADS, ITEMS, int32_t>;
  extern __shared__ __align__(16) unsigned char shared_bytes[];
  auto& storage = *reinterpret_cast<typename Sort::TempStorage*>(shared_bytes);
  __shared__ typename cub::BlockReduce<int, THREADS>::TempStorage reduce;
  float keys[ITEMS];
  int32_t values[ITEMS];
  #pragma unroll
  for (int j = 0; j < ITEMS; ++j) {
    int pos = tid * ITEMS + j;
    int token = pos < count ? ids[static_cast<int64_t>(head) * cap + pos] : -1;
    bool valid = token >= 0 && !is_selected<TAGGED>(flags, epochs, batch * 8 + group, length, token);
    keys[j] = valid ? static_cast<float>(scores[static_cast<int64_t>(head) * length + token]) : -FLT_MAX;
    values[j] = valid ? token : -1;
  }
  Sort(storage).SortDescending(keys, values);
  int local_count = 0;
  #pragma unroll
  for (int j = 0; j < ITEMS; ++j) {
    int rank = tid * ITEMS + j;
    if (rank < budget && values[j] >= 0) {
      ex_ids[static_cast<int64_t>(head) * budget + rank] = values[j];
      ++local_count;
    }
  }
  int selected = cub::BlockReduce<int, THREADS>(reduce).Sum(local_count);
  if (tid == 0) ex_counts[head] = selected;
}

template <typename T, int ITEMS, bool TAGGED>
void launch_blocksort(const int32_t* ids, const int32_t* counts, const T* scores,
                      const uint8_t* flags, const int32_t* epochs, int32_t* ex_ids,
                      int32_t* ex_counts, int batch, int length, int cap, int budget, int low) {
  using Sort = cub::BlockRadixSort<float, THREADS, ITEMS, int32_t>;
  size_t smem = sizeof(typename Sort::TempStorage);
  TORCH_CHECK(smem <= 96 * 1024, "blocksort shared memory too large: ", smem);
  if (smem > 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(exception_blocksort<T, ITEMS, TAGGED>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  }
  exception_blocksort<T, ITEMS, TAGGED><<<batch * 32, THREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
      ids, counts, scores, flags, epochs, ex_ids, ex_counts, length, cap, budget, low);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__device__ inline unsigned bf16_ordered(const uint16_t* scores, int64_t index) {
  unsigned bits = scores[index];
  return (bits & 0x8000u) ? ((~bits) & 0xffffu) : (bits ^ 0x8000u);
}

__device__ inline void warp_histogram_add(int* histogram, unsigned bin) {
  unsigned active = __activemask();
  unsigned peers = __match_any_sync(active, bin);
  int lane = threadIdx.x & 31;
  if (lane == __ffs(peers) - 1) atomicAdd(histogram + bin, __popc(peers));
}

__global__ void exception_histogram_bf16(const int32_t* __restrict__ ids,
                                         const int32_t* __restrict__ counts,
                                         const uint16_t* __restrict__ scores,
                                         const int32_t* __restrict__ flags,
                                         const int32_t* __restrict__ epochs,
                                         int32_t* __restrict__ ex_ids,
                                         int32_t* __restrict__ ex_counts,
                                         int length, int cap, int budget) {
  int head = blockIdx.x, batch = head / 32, group = (head % 32) / 4, tid = threadIdx.x;
  int group_row = batch * 8 + group;
  __shared__ int histogram[256];
  __shared__ int residual, high_bin, cutoff, remaining, cutoff_count, token_low, token_high;
  __shared__ typename cub::BlockReduce<int, THREADS>::TempStorage reduce;
  histogram[tid] = 0;
  if (tid == 0) { residual = 0; ex_counts[head] = 0; }
  __syncthreads();
  for (int pos = tid; pos < counts[head]; pos += THREADS) {
    int token = ids[static_cast<int64_t>(head) * cap + pos];
    if (flags[static_cast<int64_t>(group_row) * length + token] == epochs[group_row]) continue;
    unsigned key = bf16_ordered(scores, static_cast<int64_t>(head) * length + token);
    warp_histogram_add(histogram, key >> 8);
  }
  __syncthreads();
  if (tid == 0) {
    for (int bin = 0; bin < 256; ++bin) residual += histogram[bin];
  }
  __syncthreads();
  if (residual <= budget) {
    for (int pos = tid; pos < counts[head]; pos += THREADS) {
      int token = ids[static_cast<int64_t>(head) * cap + pos];
      if (flags[static_cast<int64_t>(group_row) * length + token] == epochs[group_row]) continue;
      int rank = atomicAdd(ex_counts + head, 1);
      ex_ids[static_cast<int64_t>(head) * budget + rank] = token;
    }
    return;
  }
  if (tid == 0) {
    int remain = budget;
    for (int bin = 255; bin >= 0; --bin) {
      if (histogram[bin] >= remain) { high_bin = bin; remaining = remain; break; }
      remain -= histogram[bin];
    }
  }
  __syncthreads();
  histogram[tid] = 0;
  __syncthreads();
  for (int pos = tid; pos < counts[head]; pos += THREADS) {
    int token = ids[static_cast<int64_t>(head) * cap + pos];
    if (flags[static_cast<int64_t>(group_row) * length + token] == epochs[group_row]) continue;
    unsigned key = bf16_ordered(scores, static_cast<int64_t>(head) * length + token);
    if ((key >> 8) == static_cast<unsigned>(high_bin)) warp_histogram_add(histogram, key & 255);
  }
  __syncthreads();
  if (tid == 0) {
    int remain = remaining;
    for (int bin = 255; bin >= 0; --bin) {
      if (histogram[bin] >= remain) {
        cutoff = (high_bin << 8) | bin;
        remaining = remain;
        cutoff_count = histogram[bin];
        break;
      }
      remain -= histogram[bin];
    }
  }
  __syncthreads();
  if (tid == 0) { token_low = 0; token_high = length - 1; }
  __syncthreads();
  if (cutoff_count > remaining) {
    while (token_low < token_high) {
      int mid = token_low + (token_high - token_low) / 2;
      int local = 0;
      for (int pos = tid; pos < counts[head]; pos += THREADS) {
        int token = ids[static_cast<int64_t>(head) * cap + pos];
        if (token > mid || flags[static_cast<int64_t>(group_row) * length + token] == epochs[group_row]) continue;
        unsigned key = bf16_ordered(scores, static_cast<int64_t>(head) * length + token);
        local += key == static_cast<unsigned>(cutoff);
      }
      int tie_prefix = cub::BlockReduce<int, THREADS>(reduce).Sum(local);
      if (tid == 0) {
        if (tie_prefix >= remaining) token_high = mid;
        else token_low = mid + 1;
      }
      __syncthreads();
    }
  }
  for (int pos = tid; pos < counts[head]; pos += THREADS) {
    int token = ids[static_cast<int64_t>(head) * cap + pos];
    if (flags[static_cast<int64_t>(group_row) * length + token] == epochs[group_row]) continue;
    unsigned key = bf16_ordered(scores, static_cast<int64_t>(head) * length + token);
    if (key > static_cast<unsigned>(cutoff)) {
      int rank = atomicAdd(ex_counts + head, 1);
      ex_ids[static_cast<int64_t>(head) * budget + rank] = token;
    }
  }
  __syncthreads();
  for (int pos = tid; pos < counts[head]; pos += THREADS) {
    int token = ids[static_cast<int64_t>(head) * cap + pos];
    if (flags[static_cast<int64_t>(group_row) * length + token] == epochs[group_row]) continue;
    unsigned key = bf16_ordered(scores, static_cast<int64_t>(head) * length + token);
    if (key == static_cast<unsigned>(cutoff)) {
      if (cutoff_count <= remaining || token <= token_low) {
        int rank = atomicAdd(ex_counts + head, 1);
        ex_ids[static_cast<int64_t>(head) * budget + rank] = token;
      }
    }
  }
}

void check(const at::Tensor& x) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "CUDA contiguous tensors required");
}
}  // namespace

void group(at::Tensor ids, at::Tensor counts, at::Tensor mask, at::Tensor scores,
           at::Tensor group_ids, at::Tensor group_counts, at::Tensor flags,
           at::Tensor scratch, double threshold) {
  for (const auto& x : {ids, counts, mask, scores, group_ids, group_counts, flags, scratch}) check(x);
  int batch = ids.size(0), cap = ids.size(2), length = mask.size(2);
  TORCH_CHECK(ids.scalar_type() == at::kInt && counts.scalar_type() == at::kInt &&
              mask.scalar_type() == at::kBool && flags.scalar_type() == at::kByte &&
              scratch.scalar_type() == at::kFloat, "invalid input dtype");
  TORCH_CHECK(ids.dim() == 3 && ids.size(1) == 32 && mask.sizes() == scores.sizes(), "invalid shapes");
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, scores.scalar_type(), "gas_group_refine", [&] {
    group_refine<scalar_t, false, false, false, false, false><<<batch * 8, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        ids.data_ptr<int32_t>(), counts.data_ptr<int32_t>(), mask.data_ptr<bool>(), scores.data_ptr<scalar_t>(),
        group_ids.data_ptr<int32_t>(), group_counts.data_ptr<int32_t>(), flags.data_ptr<uint8_t>(), nullptr,
        scratch.data_ptr<float>(), length, cap, 0, static_cast<float>(threshold));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exception(at::Tensor ids, at::Tensor counts, at::Tensor scores, at::Tensor flags,
               at::Tensor ex_ids, at::Tensor ex_counts) {
  for (const auto& x : {ids, counts, scores, flags, ex_ids, ex_counts}) check(x);
  int batch = ids.size(0), cap = ids.size(2), length = scores.size(2), budget = ex_ids.size(2);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, scores.scalar_type(), "gas_exceptions", [&] {
    exceptions<scalar_t, false><<<batch * 32, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        ids.data_ptr<int32_t>(), counts.data_ptr<int32_t>(), scores.data_ptr<scalar_t>(),
        flags.data_ptr<uint8_t>(), nullptr, ex_ids.data_ptr<int32_t>(), ex_counts.data_ptr<int32_t>(), length, cap, budget);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exception_tiled(at::Tensor ids, at::Tensor counts, at::Tensor scores, at::Tensor flags,
                     at::Tensor ex_ids, at::Tensor ex_counts, at::Tensor tmp_scores, at::Tensor tmp_ids) {
  for (const auto& x : {ids, counts, scores, flags, ex_ids, ex_counts, tmp_scores, tmp_ids}) check(x);
  int batch = ids.size(0), cap = ids.size(2), length = scores.size(2), budget = ex_ids.size(2);
  int tiles = (cap + TILE_SIZE - 1) / TILE_SIZE;
  TORCH_CHECK(budget <= TILE_KEEP && tiles * TILE_KEEP <= THREADS, "unsupported exception tile configuration");
  TORCH_CHECK(tmp_scores.scalar_type() == at::kFloat && tmp_ids.scalar_type() == at::kInt &&
              tmp_scores.numel() == batch * 32 * tiles * TILE_KEEP && tmp_ids.numel() == tmp_scores.numel(),
              "invalid exception scratch");
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, scores.scalar_type(), "gas_exception_tiles", [&] {
    exception_tiles<scalar_t, false><<<dim3(batch * 32, tiles), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        ids.data_ptr<int32_t>(), counts.data_ptr<int32_t>(), scores.data_ptr<scalar_t>(), flags.data_ptr<uint8_t>(), nullptr,
        tmp_scores.data_ptr<float>(), tmp_ids.data_ptr<int32_t>(), length, cap, tiles);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  exception_merge<<<batch * 32, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
      tmp_scores.data_ptr<float>(), tmp_ids.data_ptr<int32_t>(), ex_ids.data_ptr<int32_t>(),
      ex_counts.data_ptr<int32_t>(), tiles, budget);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void group_tagged(at::Tensor ids, at::Tensor counts, at::Tensor mask, at::Tensor scores,
                  at::Tensor group_ids, at::Tensor group_counts, at::Tensor flags,
                  at::Tensor epochs, at::Tensor scratch, double threshold) {
  for (const auto& x : {ids, counts, mask, scores, group_ids, group_counts, flags, epochs, scratch}) check(x);
  TORCH_CHECK(flags.scalar_type() == at::kInt && epochs.scalar_type() == at::kInt, "int32 tags required");
  int batch = ids.size(0), cap = ids.size(2), length = mask.size(2);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, scores.scalar_type(), "gas_group_tagged", [&] {
    group_refine<scalar_t, true, false, false, false, false><<<batch * 8, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        ids.data_ptr<int32_t>(), counts.data_ptr<int32_t>(), mask.data_ptr<bool>(), scores.data_ptr<scalar_t>(),
        group_ids.data_ptr<int32_t>(), group_counts.data_ptr<int32_t>(), reinterpret_cast<uint8_t*>(flags.data_ptr<int32_t>()),
        epochs.data_ptr<int32_t>(), scratch.data_ptr<float>(), length, cap, 0, static_cast<float>(threshold));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void group_tagged_shared(at::Tensor ids, at::Tensor counts, at::Tensor mask, at::Tensor scores,
                         at::Tensor group_ids, at::Tensor group_counts, at::Tensor flags,
                         at::Tensor epochs, at::Tensor scratch, double threshold) {
  for (const auto& x : {ids, counts, mask, scores, group_ids, group_counts, flags, epochs, scratch}) check(x);
  TORCH_CHECK(flags.scalar_type() == at::kInt && epochs.scalar_type() == at::kInt, "int32 tags required");
  int batch = ids.size(0), cap = ids.size(2), length = mask.size(2);
  int shared_cap = std::min(4 * cap, 16384);
  size_t shared_bytes = static_cast<size_t>(shared_cap) * sizeof(float);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, scores.scalar_type(), "gas_group_tagged_shared", [&] {
    if (shared_bytes > 48 * 1024) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(group_refine<scalar_t, true, true, false, false, false>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
    }
    group_refine<scalar_t, true, true, false, false, false><<<batch * 8, THREADS, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        ids.data_ptr<int32_t>(), counts.data_ptr<int32_t>(), mask.data_ptr<bool>(), scores.data_ptr<scalar_t>(),
        group_ids.data_ptr<int32_t>(), group_counts.data_ptr<int32_t>(), reinterpret_cast<uint8_t*>(flags.data_ptr<int32_t>()),
        epochs.data_ptr<int32_t>(), scratch.data_ptr<float>(), length, cap, shared_cap, static_cast<float>(threshold));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void group_tagged_dense(at::Tensor ids, at::Tensor counts, at::Tensor mask, at::Tensor scores,
                        at::Tensor group_ids, at::Tensor group_counts, at::Tensor flags,
                        at::Tensor epochs, at::Tensor scratch, double threshold) {
  for (const auto& x : {ids, counts, mask, scores, group_ids, group_counts, flags, epochs, scratch}) check(x);
  TORCH_CHECK(flags.scalar_type() == at::kInt && epochs.scalar_type() == at::kInt &&
              scratch.numel() >= ids.size(0) * 8 * mask.size(2), "invalid dense group scratch");
  int batch = ids.size(0), cap = ids.size(2), length = mask.size(2);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, scores.scalar_type(), "gas_group_tagged_dense", [&] {
    group_refine<scalar_t, true, false, true, false, false><<<batch * 8, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        ids.data_ptr<int32_t>(), counts.data_ptr<int32_t>(), mask.data_ptr<bool>(), scores.data_ptr<scalar_t>(),
        group_ids.data_ptr<int32_t>(), group_counts.data_ptr<int32_t>(), reinterpret_cast<uint8_t*>(flags.data_ptr<int32_t>()),
        epochs.data_ptr<int32_t>(), scratch.data_ptr<float>(), length, cap, 0, static_cast<float>(threshold));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void group_tagged_fused_reduce(at::Tensor ids, at::Tensor counts, at::Tensor mask, at::Tensor scores,
                               at::Tensor group_ids, at::Tensor group_counts, at::Tensor flags,
                               at::Tensor epochs, at::Tensor scratch, double threshold) {
  for (const auto& x : {ids, counts, mask, scores, group_ids, group_counts, flags, epochs, scratch}) check(x);
  TORCH_CHECK(flags.scalar_type() == at::kInt && epochs.scalar_type() == at::kInt, "int32 tags required");
  int batch = ids.size(0), cap = ids.size(2), length = mask.size(2);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, scores.scalar_type(), "gas_group_fused_reduce", [&] {
    group_refine<scalar_t, true, false, false, true, false><<<batch * 8, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        ids.data_ptr<int32_t>(), counts.data_ptr<int32_t>(), mask.data_ptr<bool>(), scores.data_ptr<scalar_t>(),
        group_ids.data_ptr<int32_t>(), group_counts.data_ptr<int32_t>(), reinterpret_cast<uint8_t*>(flags.data_ptr<int32_t>()),
        epochs.data_ptr<int32_t>(), scratch.data_ptr<float>(), length, cap, 0, static_cast<float>(threshold));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void group_tagged_normalized(at::Tensor ids, at::Tensor counts, at::Tensor mask, at::Tensor scores,
                             at::Tensor group_ids, at::Tensor group_counts, at::Tensor flags,
                             at::Tensor epochs, at::Tensor scratch, double threshold) {
  for (const auto& x : {ids, counts, mask, scores, group_ids, group_counts, flags, epochs, scratch}) check(x);
  TORCH_CHECK(flags.scalar_type() == at::kInt && epochs.scalar_type() == at::kInt, "int32 tags required");
  int batch = ids.size(0), cap = ids.size(2), length = mask.size(2);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, scores.scalar_type(), "gas_group_normalized", [&] {
    group_refine<scalar_t, true, false, false, true, false, false, false, false, false, true>
        <<<batch * 8, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        ids.data_ptr<int32_t>(), counts.data_ptr<int32_t>(), mask.data_ptr<bool>(), scores.data_ptr<scalar_t>(),
        group_ids.data_ptr<int32_t>(), group_counts.data_ptr<int32_t>(), reinterpret_cast<uint8_t*>(flags.data_ptr<int32_t>()),
        epochs.data_ptr<int32_t>(), scratch.data_ptr<float>(), length, cap, 0, static_cast<float>(threshold));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void group_tagged_normalized_shared(at::Tensor ids, at::Tensor counts, at::Tensor mask, at::Tensor scores,
                                    at::Tensor group_ids, at::Tensor group_counts, at::Tensor flags,
                                    at::Tensor epochs, at::Tensor scratch, double threshold) {
  for (const auto& x : {ids, counts, mask, scores, group_ids, group_counts, flags, epochs, scratch}) check(x);
  TORCH_CHECK(flags.scalar_type() == at::kInt && epochs.scalar_type() == at::kInt, "int32 tags required");
  int batch = ids.size(0), cap = ids.size(2), length = mask.size(2);
  int shared_cap = std::min(4 * cap, 18432);
  size_t shared_bytes = static_cast<size_t>(shared_cap) * sizeof(float);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, scores.scalar_type(), "gas_group_normalized_shared", [&] {
    if (shared_bytes > 48 * 1024) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          group_refine<scalar_t, true, true, false, true, false, false, false, false, false, true>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
    }
    group_refine<scalar_t, true, true, false, true, false, false, false, false, false, true>
        <<<batch * 8, THREADS, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        ids.data_ptr<int32_t>(), counts.data_ptr<int32_t>(), mask.data_ptr<bool>(), scores.data_ptr<scalar_t>(),
        group_ids.data_ptr<int32_t>(), group_counts.data_ptr<int32_t>(), reinterpret_cast<uint8_t*>(flags.data_ptr<int32_t>()),
        epochs.data_ptr<int32_t>(), scratch.data_ptr<float>(), length, cap, shared_cap, static_cast<float>(threshold));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void group_tagged_normalized_fast_scores(at::Tensor ids, at::Tensor counts, at::Tensor mask, at::Tensor scores,
                                         at::Tensor group_ids, at::Tensor group_counts, at::Tensor flags,
                                         at::Tensor epochs, at::Tensor scratch, double threshold) {
  for (const auto& x : {ids, counts, mask, scores, group_ids, group_counts, flags, epochs, scratch}) check(x);
  TORCH_CHECK(flags.scalar_type() == at::kInt && epochs.scalar_type() == at::kInt, "int32 tags required");
  int batch = ids.size(0), cap = ids.size(2), length = mask.size(2);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, scores.scalar_type(), "gas_group_normalized_fast_scores", [&] {
    group_refine<scalar_t, true, false, false, true, false, false, true, false, false, true>
        <<<batch * 8, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        ids.data_ptr<int32_t>(), counts.data_ptr<int32_t>(), mask.data_ptr<bool>(), scores.data_ptr<scalar_t>(),
        group_ids.data_ptr<int32_t>(), group_counts.data_ptr<int32_t>(), reinterpret_cast<uint8_t*>(flags.data_ptr<int32_t>()),
        epochs.data_ptr<int32_t>(), scratch.data_ptr<float>(), length, cap, 0, static_cast<float>(threshold));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void group_tagged_normalized_prepared(at::Tensor ids, at::Tensor counts, at::Tensor mask, at::Tensor scores,
                                      at::Tensor group_ids, at::Tensor group_counts, at::Tensor flags,
                                      at::Tensor epochs, at::Tensor scratch, double threshold) {
  for (const auto& x : {ids, counts, mask, scores, group_ids, group_counts, flags, epochs, scratch}) check(x);
  TORCH_CHECK(flags.scalar_type() == at::kInt && epochs.scalar_type() == at::kInt, "int32 tags required");
  int batch = ids.size(0), cap = ids.size(2), length = mask.size(2);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, scores.scalar_type(), "gas_group_normalized_prepared", [&] {
    prepare_group_scores<scalar_t><<<batch * 32, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        ids.data_ptr<int32_t>(), counts.data_ptr<int32_t>(), mask.data_ptr<bool>(),
        scores.data_ptr<scalar_t>(), scratch.data_ptr<float>(), length, cap);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    group_refine<scalar_t, true, false, false, true, false, false, false, false, false, true, true>
        <<<batch * 8, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        ids.data_ptr<int32_t>(), counts.data_ptr<int32_t>(), mask.data_ptr<bool>(), scores.data_ptr<scalar_t>(),
        group_ids.data_ptr<int32_t>(), group_counts.data_ptr<int32_t>(), reinterpret_cast<uint8_t*>(flags.data_ptr<int32_t>()),
        epochs.data_ptr<int32_t>(), scratch.data_ptr<float>(), length, cap, 0, static_cast<float>(threshold));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void group_tagged_fused_exception(at::Tensor ids, at::Tensor counts, at::Tensor mask, at::Tensor scores,
                                  at::Tensor group_ids, at::Tensor group_counts, at::Tensor flags,
                                  at::Tensor epochs, at::Tensor scratch, at::Tensor ex_ids,
                                  at::Tensor ex_counts, double threshold) {
  for (const auto& x : {ids, counts, mask, scores, group_ids, group_counts, flags, epochs, scratch, ex_ids, ex_counts}) check(x);
  TORCH_CHECK(flags.scalar_type() == at::kInt && epochs.scalar_type() == at::kInt &&
              ex_ids.scalar_type() == at::kInt && ex_ids.size(2) == 16 && ex_counts.scalar_type() == at::kInt,
              "int32 tags and a 16-token exception buffer required");
  int batch = ids.size(0), cap = ids.size(2), length = mask.size(2);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, scores.scalar_type(), "gas_group_fused_exception", [&] {
    group_refine<scalar_t, true, false, false, true, false, true><<<batch * 8, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        ids.data_ptr<int32_t>(), counts.data_ptr<int32_t>(), mask.data_ptr<bool>(), scores.data_ptr<scalar_t>(),
        group_ids.data_ptr<int32_t>(), group_counts.data_ptr<int32_t>(), reinterpret_cast<uint8_t*>(flags.data_ptr<int32_t>()),
        epochs.data_ptr<int32_t>(), scratch.data_ptr<float>(), length, cap, 0, static_cast<float>(threshold),
        ex_ids.data_ptr<int32_t>(), ex_counts.data_ptr<int32_t>());
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void group_tagged_fast_scores(at::Tensor ids, at::Tensor counts, at::Tensor mask, at::Tensor scores,
                              at::Tensor group_ids, at::Tensor group_counts, at::Tensor flags,
                              at::Tensor epochs, at::Tensor scratch, double threshold) {
  for (const auto& x : {ids, counts, mask, scores, group_ids, group_counts, flags, epochs, scratch}) check(x);
  TORCH_CHECK(flags.scalar_type() == at::kInt && epochs.scalar_type() == at::kInt, "int32 tags required");
  int batch = ids.size(0), cap = ids.size(2), length = mask.size(2);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, scores.scalar_type(), "gas_group_fast_scores", [&] {
    group_refine<scalar_t, true, false, false, true, false, false, true><<<batch * 8, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        ids.data_ptr<int32_t>(), counts.data_ptr<int32_t>(), mask.data_ptr<bool>(), scores.data_ptr<scalar_t>(),
        group_ids.data_ptr<int32_t>(), group_counts.data_ptr<int32_t>(), reinterpret_cast<uint8_t*>(flags.data_ptr<int32_t>()),
        epochs.data_ptr<int32_t>(), scratch.data_ptr<float>(), length, cap, 0, static_cast<float>(threshold));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void group_tagged_fused_mass(at::Tensor ids, at::Tensor counts, at::Tensor mask, at::Tensor scores,
                             at::Tensor group_ids, at::Tensor group_counts, at::Tensor flags,
                             at::Tensor epochs, at::Tensor scratch, double threshold) {
  for (const auto& x : {ids, counts, mask, scores, group_ids, group_counts, flags, epochs, scratch}) check(x);
  TORCH_CHECK(flags.scalar_type() == at::kInt && epochs.scalar_type() == at::kInt, "int32 tags required");
  int batch = ids.size(0), cap = ids.size(2), length = mask.size(2);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, scores.scalar_type(), "gas_group_fused_mass", [&] {
    group_refine<scalar_t, true, false, false, true, false, false, false, true><<<batch * 8, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        ids.data_ptr<int32_t>(), counts.data_ptr<int32_t>(), mask.data_ptr<bool>(), scores.data_ptr<scalar_t>(),
        group_ids.data_ptr<int32_t>(), group_counts.data_ptr<int32_t>(), reinterpret_cast<uint8_t*>(flags.data_ptr<int32_t>()),
        epochs.data_ptr<int32_t>(), scratch.data_ptr<float>(), length, cap, 0, static_cast<float>(threshold));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void group_head_tagged(at::Tensor ids, at::Tensor counts, at::Tensor head_flags, at::Tensor head_epochs,
                       at::Tensor scores, at::Tensor group_ids, at::Tensor group_counts, at::Tensor flags,
                       at::Tensor epochs, at::Tensor scratch, double threshold) {
  for (const auto& x : {ids, counts, head_flags, head_epochs, scores, group_ids, group_counts, flags, epochs, scratch}) check(x);
  TORCH_CHECK(head_flags.scalar_type() == at::kInt && head_epochs.scalar_type() == at::kInt &&
              flags.scalar_type() == at::kInt && epochs.scalar_type() == at::kInt, "int32 tags required");
  int batch = ids.size(0), cap = ids.size(2), length = head_flags.size(2);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, scores.scalar_type(), "gas_group_head_tagged", [&] {
    group_refine<scalar_t, true, false, false, true, false, false, false, false, true><<<batch * 8, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        ids.data_ptr<int32_t>(), counts.data_ptr<int32_t>(), reinterpret_cast<const bool*>(head_flags.data_ptr<int32_t>()),
        scores.data_ptr<scalar_t>(), group_ids.data_ptr<int32_t>(), group_counts.data_ptr<int32_t>(),
        reinterpret_cast<uint8_t*>(flags.data_ptr<int32_t>()), epochs.data_ptr<int32_t>(), scratch.data_ptr<float>(),
        length, cap, 0, static_cast<float>(threshold), nullptr, nullptr, head_epochs.data_ptr<int32_t>());
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void group_tagged_warp_reduce(at::Tensor ids, at::Tensor counts, at::Tensor mask, at::Tensor scores,
                              at::Tensor group_ids, at::Tensor group_counts, at::Tensor flags,
                              at::Tensor epochs, at::Tensor scratch, double threshold) {
  for (const auto& x : {ids, counts, mask, scores, group_ids, group_counts, flags, epochs, scratch}) check(x);
  TORCH_CHECK(flags.scalar_type() == at::kInt && epochs.scalar_type() == at::kInt, "int32 tags required");
  int batch = ids.size(0), cap = ids.size(2), length = mask.size(2);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, scores.scalar_type(), "gas_group_warp_reduce", [&] {
    group_refine<scalar_t, true, false, false, true, true><<<batch * 8, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        ids.data_ptr<int32_t>(), counts.data_ptr<int32_t>(), mask.data_ptr<bool>(), scores.data_ptr<scalar_t>(),
        group_ids.data_ptr<int32_t>(), group_counts.data_ptr<int32_t>(), reinterpret_cast<uint8_t*>(flags.data_ptr<int32_t>()),
        epochs.data_ptr<int32_t>(), scratch.data_ptr<float>(), length, cap, 0, static_cast<float>(threshold));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exception_tiled_tagged(at::Tensor ids, at::Tensor counts, at::Tensor scores, at::Tensor flags,
                            at::Tensor epochs, at::Tensor ex_ids, at::Tensor ex_counts,
                            at::Tensor tmp_scores, at::Tensor tmp_ids) {
  for (const auto& x : {ids, counts, scores, flags, epochs, ex_ids, ex_counts, tmp_scores, tmp_ids}) check(x);
  int batch = ids.size(0), cap = ids.size(2), length = scores.size(2), budget = ex_ids.size(2);
  int tiles = (cap + TILE_SIZE - 1) / TILE_SIZE;
  TORCH_CHECK(flags.scalar_type() == at::kInt && epochs.scalar_type() == at::kInt && budget <= TILE_KEEP &&
              tiles * TILE_KEEP <= THREADS && tmp_scores.scalar_type() == at::kFloat &&
              tmp_ids.scalar_type() == at::kInt && tmp_scores.numel() == batch * 32 * tiles * TILE_KEEP &&
              tmp_ids.numel() == tmp_scores.numel(), "invalid tagged exception scratch");
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, scores.scalar_type(), "gas_exception_tiles_tagged", [&] {
    exception_tiles<scalar_t, true><<<dim3(batch * 32, tiles), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        ids.data_ptr<int32_t>(), counts.data_ptr<int32_t>(), scores.data_ptr<scalar_t>(),
        reinterpret_cast<uint8_t*>(flags.data_ptr<int32_t>()), epochs.data_ptr<int32_t>(),
        tmp_scores.data_ptr<float>(), tmp_ids.data_ptr<int32_t>(), length, cap, tiles);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  exception_merge<<<batch * 32, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
      tmp_scores.data_ptr<float>(), tmp_ids.data_ptr<int32_t>(), ex_ids.data_ptr<int32_t>(),
      ex_counts.data_ptr<int32_t>(), tiles, budget);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exception_blocksort_tagged(at::Tensor ids, at::Tensor counts, at::Tensor scores, at::Tensor flags,
                                at::Tensor epochs, at::Tensor ex_ids, at::Tensor ex_counts) {
  for (const auto& x : {ids, counts, scores, flags, epochs, ex_ids, ex_counts}) check(x);
  int batch = ids.size(0), cap = ids.size(2), length = scores.size(2), budget = ex_ids.size(2);
  TORCH_CHECK(flags.scalar_type() == at::kInt && epochs.scalar_type() == at::kInt &&
              cap <= 8192 && budget <= 16, "unsupported tagged blocksort configuration");
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, scores.scalar_type(), "gas_exception_blocksort", [&] {
    auto ids_ptr = ids.data_ptr<int32_t>();
    auto counts_ptr = counts.data_ptr<int32_t>();
    auto score_ptr = scores.data_ptr<scalar_t>();
    auto flags_ptr = reinterpret_cast<uint8_t*>(flags.data_ptr<int32_t>());
    auto epoch_ptr = epochs.data_ptr<int32_t>();
    auto ex_ptr = ex_ids.data_ptr<int32_t>();
    auto ec_ptr = ex_counts.data_ptr<int32_t>();
    launch_blocksort<scalar_t, 1, true>(ids_ptr, counts_ptr, score_ptr, flags_ptr, epoch_ptr, ex_ptr, ec_ptr, batch, length, cap, budget, -1);
    if (cap > 256) launch_blocksort<scalar_t, 4, true>(ids_ptr, counts_ptr, score_ptr, flags_ptr, epoch_ptr, ex_ptr, ec_ptr, batch, length, cap, budget, 256);
    if (cap > 1024) launch_blocksort<scalar_t, 16, true>(ids_ptr, counts_ptr, score_ptr, flags_ptr, epoch_ptr, ex_ptr, ec_ptr, batch, length, cap, budget, 1024);
    if (cap > 4096) launch_blocksort<scalar_t, 32, true>(ids_ptr, counts_ptr, score_ptr, flags_ptr, epoch_ptr, ex_ptr, ec_ptr, batch, length, cap, budget, 4096);
  });
}

void exception_histogram_tagged(at::Tensor ids, at::Tensor counts, at::Tensor scores, at::Tensor flags,
                                at::Tensor epochs, at::Tensor ex_ids, at::Tensor ex_counts) {
  for (const auto& x : {ids, counts, scores, flags, epochs, ex_ids, ex_counts}) check(x);
  TORCH_CHECK(scores.scalar_type() == at::kBFloat16 && flags.scalar_type() == at::kInt &&
              epochs.scalar_type() == at::kInt && ex_ids.size(2) <= 16, "BF16 scores and int32 tags required");
  int batch = ids.size(0), cap = ids.size(2), length = scores.size(2), budget = ex_ids.size(2);
  exception_histogram_bf16<<<batch * 32, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
      ids.data_ptr<int32_t>(), counts.data_ptr<int32_t>(), reinterpret_cast<const uint16_t*>(scores.data_ptr<at::BFloat16>()),
      flags.data_ptr<int32_t>(), epochs.data_ptr<int32_t>(), ex_ids.data_ptr<int32_t>(), ex_counts.data_ptr<int32_t>(),
      length, cap, budget);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("group", &group);
  m.def("exception", &exception);
  m.def("exception_tiled", &exception_tiled);
  m.def("group_tagged", &group_tagged);
  m.def("group_tagged_shared", &group_tagged_shared);
  m.def("group_tagged_dense", &group_tagged_dense);
  m.def("group_tagged_fused_reduce", &group_tagged_fused_reduce);
  m.def("group_tagged_normalized", &group_tagged_normalized);
  m.def("group_tagged_normalized_shared", &group_tagged_normalized_shared);
  m.def("group_tagged_normalized_fast_scores", &group_tagged_normalized_fast_scores);
  m.def("group_tagged_normalized_prepared", &group_tagged_normalized_prepared);
  m.def("group_tagged_fused_exception", &group_tagged_fused_exception);
  m.def("group_tagged_fast_scores", &group_tagged_fast_scores);
  m.def("group_tagged_fused_mass", &group_tagged_fused_mass);
  m.def("group_head_tagged", &group_head_tagged);
  m.def("group_tagged_warp_reduce", &group_tagged_warp_reduce);
  m.def("exception_tiled_tagged", &exception_tiled_tagged);
  m.def("exception_blocksort_tagged", &exception_blocksort_tagged);
  m.def("exception_histogram_tagged", &exception_histogram_tagged);
}
