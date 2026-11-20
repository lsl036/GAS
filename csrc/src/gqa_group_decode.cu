#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr int kBlockThreads = 256;
constexpr int kMaxGroupSize = 8;

__device__ inline float LoadScore(const float* __restrict__ estimated_weights,
                                  int64_t offset) {
  return estimated_weights[offset];
}

__device__ inline bool LoadMask(const bool* __restrict__ mask, int64_t offset) {
  return mask[offset];
}

__global__ void BuildGroupMaxSharedIndicesKernel(const float* __restrict__ estimated_weights,
                                                 const bool* __restrict__ candidate_mask,
                                                 int32_t* __restrict__ group_indices,
                                                 int32_t* __restrict__ group_counts,
                                                 int32_t batch_size,
                                                 int32_t num_heads,
                                                 int32_t num_kv_heads,
                                                 int32_t seq_len,
                                                 int32_t max_group_selected,
                                                 float threshold,
                                                 int32_t group_size) {
  const int32_t kv_head = blockIdx.x;
  const int32_t batch_idx = blockIdx.y;
  const int32_t tid = threadIdx.x;
  if (batch_idx >= batch_size || kv_head >= num_kv_heads) {
    return;
  }

  extern __shared__ float smem[];
  float* scores = smem;
  int32_t* tokens = reinterpret_cast<int32_t*>(scores + max_group_selected);
  float* scratch = reinterpret_cast<float*>(tokens + max_group_selected);

  const int32_t head_begin = kv_head * group_size;
  const int64_t batch_head_base =
      static_cast<int64_t>(batch_idx) * num_heads * seq_len;
  const int64_t group_out_base =
      (static_cast<int64_t>(batch_idx) * num_kv_heads + kv_head) * max_group_selected;

  for (int32_t pos = tid; pos < max_group_selected; pos += blockDim.x) {
    group_indices[group_out_base + pos] = -1;
  }
  if (tid == 0) {
    group_counts[batch_idx * num_kv_heads + kv_head] = 0;
  }
  __syncthreads();

  float local_max = -FLT_MAX;
  for (int32_t token = tid; token < seq_len; token += blockDim.x) {
    bool valid = false;
    float score = -FLT_MAX;
    for (int32_t h = 0; h < group_size; ++h) {
      const int32_t q_head = head_begin + h;
      const int64_t offset = batch_head_base + static_cast<int64_t>(q_head) * seq_len + token;
      if (LoadMask(candidate_mask, offset)) {
        valid = true;
        score = fmaxf(score, LoadScore(estimated_weights, offset));
      }
    }
    if (valid) {
      local_max = fmaxf(local_max, score);
    }
  }
  scratch[tid] = local_max;
  __syncthreads();
  for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
    }
    __syncthreads();
  }
  const float max_score = scratch[0];
  __syncthreads();

  float local_sum = 0.0f;
  for (int32_t token = tid; token < seq_len; token += blockDim.x) {
    bool valid = false;
    float score = -FLT_MAX;
    for (int32_t h = 0; h < group_size; ++h) {
      const int32_t q_head = head_begin + h;
      const int64_t offset = batch_head_base + static_cast<int64_t>(q_head) * seq_len + token;
      if (LoadMask(candidate_mask, offset)) {
        valid = true;
        score = fmaxf(score, LoadScore(estimated_weights, offset));
      }
    }
    if (valid) {
      local_sum += expf(score - max_score);
    }
  }
  scratch[tid] = local_sum;
  __syncthreads();
  for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  const float denom = scratch[0];
  if (denom <= 0.0f || max_score == -FLT_MAX) {
    return;
  }

  float local_threshold = 0.0f;
  if (tid == 0) {
    float cumulative_prob = 0.0f;
    for (int32_t rank = 0; rank < max_group_selected; ++rank) {
      float best_prob = -1.0f;
      int32_t best_token = -1;
      for (int32_t token = 0; token < seq_len; ++token) {
        bool already_selected = false;
        for (int32_t prev = 0; prev < rank; ++prev) {
          if (group_indices[group_out_base + prev] == token) {
            already_selected = true;
            break;
          }
        }
        if (already_selected) {
          continue;
        }

        bool valid = false;
        float score = -FLT_MAX;
        for (int32_t h = 0; h < group_size; ++h) {
          const int32_t q_head = head_begin + h;
          const int64_t offset =
              batch_head_base + static_cast<int64_t>(q_head) * seq_len + token;
          if (LoadMask(candidate_mask, offset)) {
            valid = true;
            score = fmaxf(score, LoadScore(estimated_weights, offset));
          }
        }
        if (!valid) {
          continue;
        }
        const float prob = expf(score - max_score) / denom;
        if (prob > best_prob) {
          best_prob = prob;
          best_token = token;
        }
      }
      if (best_token < 0) {
        break;
      }
      group_indices[group_out_base + rank] = best_token;
      group_counts[batch_idx * num_kv_heads + kv_head] = rank + 1;
      cumulative_prob += best_prob;
      if (cumulative_prob >= threshold) {
        break;
      }
    }
  }
}

__global__ void BuildPerHeadExceptionIndicesKernel(const float* __restrict__ estimated_weights,
                                                   const bool* __restrict__ candidate_mask,
                                                   const int32_t* __restrict__ group_indices,
                                                   int32_t* __restrict__ exception_indices,
                                                   int32_t* __restrict__ exception_counts,
                                                   int32_t batch_size,
                                                   int32_t num_heads,
                                                   int32_t num_kv_heads,
                                                   int32_t seq_len,
                                                   int32_t max_group_selected,
                                                   int32_t exception_budget,
                                                   int32_t group_size) {
  const int32_t q_head = blockIdx.x;
  const int32_t batch_idx = blockIdx.y;
  const int32_t tid = threadIdx.x;
  if (batch_idx >= batch_size || q_head >= num_heads) {
    return;
  }

  const int32_t kv_head = q_head / group_size;
  const int64_t head_base =
      (static_cast<int64_t>(batch_idx) * num_heads + q_head) * seq_len;
  const int64_t group_base =
      (static_cast<int64_t>(batch_idx) * num_kv_heads + kv_head) * max_group_selected;
  const int64_t exception_base =
      (static_cast<int64_t>(batch_idx) * num_heads + q_head) * exception_budget;

  for (int32_t pos = tid; pos < exception_budget; pos += blockDim.x) {
    exception_indices[exception_base + pos] = -1;
  }
  if (tid == 0) {
    exception_counts[batch_idx * num_heads + q_head] = 0;
    for (int32_t rank = 0; rank < exception_budget; ++rank) {
      float best_score = -FLT_MAX;
      int32_t best_token = -1;
      for (int32_t token = 0; token < seq_len; ++token) {
        if (!LoadMask(candidate_mask, head_base + token)) {
          continue;
        }
        bool in_group = false;
        for (int32_t group_pos = 0; group_pos < max_group_selected; ++group_pos) {
          const int32_t group_token = group_indices[group_base + group_pos];
          if (group_token < 0) {
            break;
          }
          if (group_token == token) {
            in_group = true;
            break;
          }
        }
        if (in_group) {
          continue;
        }
        bool already_selected = false;
        for (int32_t prev = 0; prev < rank; ++prev) {
          if (exception_indices[exception_base + prev] == token) {
            already_selected = true;
            break;
          }
        }
        if (already_selected) {
          continue;
        }
        const float score = LoadScore(estimated_weights, head_base + token);
        if (score > best_score) {
          best_score = score;
          best_token = token;
        }
      }
      if (best_token < 0) {
        break;
      }
      exception_indices[exception_base + rank] = best_token;
      exception_counts[batch_idx * num_heads + q_head] = rank + 1;
    }
  }
}

__global__ void CompactGroupMaskKernel(const bool* __restrict__ shared_mask,
                                       int32_t* __restrict__ group_indices,
                                       int32_t* __restrict__ group_counts,
                                       int32_t batch_size,
                                       int32_t num_heads,
                                       int32_t num_kv_heads,
                                       int32_t seq_len,
                                       int32_t max_group_selected,
                                       int32_t group_size) {
  const int32_t kv_head = blockIdx.x;
  const int32_t batch_idx = blockIdx.y;
  const int32_t tid = threadIdx.x;
  if (batch_idx >= batch_size || kv_head >= num_kv_heads) {
    return;
  }

  const int32_t head_begin = kv_head * group_size;
  const int64_t batch_head_base =
      static_cast<int64_t>(batch_idx) * num_heads * seq_len;
  const int64_t group_out_base =
      (static_cast<int64_t>(batch_idx) * num_kv_heads + kv_head) * max_group_selected;

  for (int32_t pos = tid; pos < max_group_selected; pos += blockDim.x) {
    group_indices[group_out_base + pos] = -1;
  }
  if (tid == 0) {
    int32_t count = 0;
    for (int32_t token = 0; token < seq_len && count < max_group_selected; ++token) {
      bool selected = false;
      for (int32_t h = 0; h < group_size; ++h) {
        const int32_t q_head = head_begin + h;
        if (LoadMask(shared_mask,
                     batch_head_base + static_cast<int64_t>(q_head) * seq_len + token)) {
          selected = true;
          break;
        }
      }
      if (selected) {
        group_indices[group_out_base + count] = token;
        ++count;
      }
    }
    group_counts[batch_idx * num_kv_heads + kv_head] = count;
  }
}

__global__ void CompactExceptionMaskKernel(const bool* __restrict__ final_mask,
                                           const bool* __restrict__ shared_mask,
                                           int32_t* __restrict__ exception_indices,
                                           int32_t* __restrict__ exception_counts,
                                           int32_t batch_size,
                                           int32_t num_heads,
                                           int32_t seq_len,
                                           int32_t exception_budget) {
  const int32_t q_head = blockIdx.x;
  const int32_t batch_idx = blockIdx.y;
  const int32_t tid = threadIdx.x;
  if (batch_idx >= batch_size || q_head >= num_heads) {
    return;
  }

  const int64_t head_base =
      (static_cast<int64_t>(batch_idx) * num_heads + q_head) * seq_len;
  const int64_t exception_base =
      (static_cast<int64_t>(batch_idx) * num_heads + q_head) * exception_budget;

  for (int32_t pos = tid; pos < exception_budget; pos += blockDim.x) {
    exception_indices[exception_base + pos] = -1;
  }
  if (tid == 0) {
    int32_t count = 0;
    for (int32_t token = 0; token < seq_len && count < exception_budget; ++token) {
      if (LoadMask(final_mask, head_base + token) &&
          !LoadMask(shared_mask, head_base + token)) {
        exception_indices[exception_base + count] = token;
        ++count;
      }
    }
    exception_counts[batch_idx * num_heads + q_head] = count;
  }
}

__global__ void GQAGroupSharedDecodeKernel(const half* __restrict__ q,
                                           const half* __restrict__ k,
                                           const half* __restrict__ v,
                                           const int32_t* __restrict__ indices,
                                           half* __restrict__ out,
                                           int32_t num_heads,
                                           int32_t num_kv_heads,
                                           int32_t seq_len,
                                           int32_t head_dim,
                                           int32_t max_selected,
                                           int32_t group_size) {
  const int32_t kv_head = blockIdx.x;
  const int32_t tid = threadIdx.x;
  if (kv_head >= num_kv_heads) {
    return;
  }

  extern __shared__ float smem[];
  float* logits = smem;
  float* scratch = logits + group_size * max_selected;

  const float scale = rsqrtf(static_cast<float>(head_dim));
  const int32_t head_begin = kv_head * group_size;

  for (int32_t selected_pos = tid; selected_pos < max_selected; selected_pos += blockDim.x) {
    const int32_t token = indices[kv_head * max_selected + selected_pos];
    float dots[kMaxGroupSize];
#pragma unroll
    for (int32_t h = 0; h < kMaxGroupSize; ++h) {
      dots[h] = 0.0f;
    }

    if (token >= 0 && token < seq_len) {
      for (int32_t dim = 0; dim < head_dim; ++dim) {
        const float k_val =
            __half2float(k[((kv_head * seq_len + token) * head_dim) + dim]);
        for (int32_t h = 0; h < group_size; ++h) {
          const int32_t q_head = head_begin + h;
          const float q_val = __half2float(q[q_head * head_dim + dim]);
          dots[h] += q_val * k_val;
        }
      }
    }

    for (int32_t h = 0; h < group_size; ++h) {
      logits[h * max_selected + selected_pos] =
          (token >= 0 && token < seq_len) ? dots[h] * scale : -FLT_MAX;
    }
  }
  __syncthreads();

  for (int32_t h = 0; h < group_size; ++h) {
    float local_max = -FLT_MAX;
    for (int32_t selected_pos = tid; selected_pos < max_selected; selected_pos += blockDim.x) {
      local_max = fmaxf(local_max, logits[h * max_selected + selected_pos]);
    }
    scratch[tid] = local_max;
    __syncthreads();
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
      }
      __syncthreads();
    }
    const float max_val = scratch[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int32_t selected_pos = tid; selected_pos < max_selected; selected_pos += blockDim.x) {
      const int32_t offset = h * max_selected + selected_pos;
      const float exp_val = expf(logits[offset] - max_val);
      logits[offset] = exp_val;
      local_sum += exp_val;
    }
    scratch[tid] = local_sum;
    __syncthreads();
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        scratch[tid] += scratch[tid + stride];
      }
      __syncthreads();
    }
    const float inv_sum = 1.0f / scratch[0];

    for (int32_t selected_pos = tid; selected_pos < max_selected; selected_pos += blockDim.x) {
      logits[h * max_selected + selected_pos] *= inv_sum;
    }
    __syncthreads();
  }

  for (int32_t dim = tid; dim < head_dim; dim += blockDim.x) {
    float acc[kMaxGroupSize];
#pragma unroll
    for (int32_t h = 0; h < kMaxGroupSize; ++h) {
      acc[h] = 0.0f;
    }

    for (int32_t selected_pos = 0; selected_pos < max_selected; ++selected_pos) {
      const int32_t token = indices[kv_head * max_selected + selected_pos];
      if (token < 0 || token >= seq_len) {
        continue;
      }
      const float v_val =
          __half2float(v[((kv_head * seq_len + token) * head_dim) + dim]);
      for (int32_t h = 0; h < group_size; ++h) {
        acc[h] += logits[h * max_selected + selected_pos] * v_val;
      }
    }

    for (int32_t h = 0; h < group_size; ++h) {
      const int32_t q_head = head_begin + h;
      out[q_head * head_dim + dim] = __float2half(acc[h]);
    }
  }
}

__global__ void BatchedGQAGroupSharedDecodeKernel(const half* __restrict__ q,
                                                  const half* __restrict__ k,
                                                  const half* __restrict__ v,
                                                  const int32_t* __restrict__ indices,
                                                  half* __restrict__ out,
                                                  int32_t batch_size,
                                                  int32_t num_heads,
                                                  int32_t num_kv_heads,
                                                  int32_t seq_len,
                                                  int32_t head_dim,
                                                  int32_t max_selected,
                                                  int32_t group_size) {
  const int32_t kv_head = blockIdx.x;
  const int32_t batch_idx = blockIdx.y;
  const int32_t tid = threadIdx.x;
  if (batch_idx >= batch_size || kv_head >= num_kv_heads) {
    return;
  }

  extern __shared__ float smem[];
  float* logits = smem;
  float* scratch = logits + group_size * max_selected;

  const float scale = rsqrtf(static_cast<float>(head_dim));
  const int32_t head_begin = kv_head * group_size;
  const int64_t q_batch_base = static_cast<int64_t>(batch_idx) * num_heads * head_dim;
  const int64_t kv_batch_base =
      static_cast<int64_t>(batch_idx) * num_kv_heads * seq_len * head_dim;
  const int64_t index_base =
      (static_cast<int64_t>(batch_idx) * num_kv_heads + kv_head) * max_selected;
  const int64_t out_batch_base = q_batch_base;

  const int32_t logits_count = group_size * max_selected;
  for (int32_t linear = tid; linear < logits_count; linear += blockDim.x) {
    const int32_t h = linear / max_selected;
    const int32_t selected_pos = linear - h * max_selected;
    const int32_t token = indices[index_base + selected_pos];
    float dot = 0.0f;
    if (token >= 0 && token < seq_len) {
      const int64_t k_base =
          kv_batch_base + (static_cast<int64_t>(kv_head) * seq_len + token) * head_dim;
      const int32_t q_head = head_begin + h;
      for (int32_t dim = 0; dim < head_dim; ++dim) {
        dot += __half2float(q[q_batch_base + q_head * head_dim + dim]) *
               __half2float(k[k_base + dim]);
      }
    }
    logits[linear] = (token >= 0 && token < seq_len) ? dot * scale : -FLT_MAX;
  }
  __syncthreads();

  for (int32_t h = 0; h < group_size; ++h) {
    float local_max = -FLT_MAX;
    for (int32_t selected_pos = tid; selected_pos < max_selected; selected_pos += blockDim.x) {
      local_max = fmaxf(local_max, logits[h * max_selected + selected_pos]);
    }
    scratch[tid] = local_max;
    __syncthreads();
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
      }
      __syncthreads();
    }
    const float max_val = scratch[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int32_t selected_pos = tid; selected_pos < max_selected; selected_pos += blockDim.x) {
      const int32_t offset = h * max_selected + selected_pos;
      const float exp_val = expf(logits[offset] - max_val);
      logits[offset] = exp_val;
      local_sum += exp_val;
    }
    scratch[tid] = local_sum;
    __syncthreads();
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        scratch[tid] += scratch[tid + stride];
      }
      __syncthreads();
    }
    const float inv_sum = 1.0f / scratch[0];

    for (int32_t selected_pos = tid; selected_pos < max_selected; selected_pos += blockDim.x) {
      logits[h * max_selected + selected_pos] *= inv_sum;
    }
    __syncthreads();
  }

  const int32_t output_count = group_size * head_dim;
  for (int32_t linear = tid; linear < output_count; linear += blockDim.x) {
    const int32_t h = linear / head_dim;
    const int32_t dim = linear - h * head_dim;
    float acc = 0.0f;
    for (int32_t selected_pos = 0; selected_pos < max_selected; ++selected_pos) {
      const int32_t token = indices[index_base + selected_pos];
      if (token < 0 || token >= seq_len) {
        continue;
      }
      const int64_t v_base =
          kv_batch_base + (static_cast<int64_t>(kv_head) * seq_len + token) * head_dim;
      const float v_val = __half2float(v[v_base + dim]);
      acc += logits[h * max_selected + selected_pos] * v_val;
    }
    const int32_t q_head = head_begin + h;
    out[out_batch_base + q_head * head_dim + dim] = __float2half(acc);
  }
}

__global__ void BatchedGQAGroupExceptionDecodeKernel(const half* __restrict__ q,
                                                     const half* __restrict__ k,
                                                     const half* __restrict__ v,
                                                     const int32_t* __restrict__ group_indices,
                                                     const int32_t* __restrict__ exception_indices,
                                                     half* __restrict__ out,
                                                     int32_t batch_size,
                                                     int32_t num_heads,
                                                     int32_t num_kv_heads,
                                                     int32_t seq_len,
                                                     int32_t head_dim,
                                                     int32_t max_group_selected,
                                                     int32_t exception_budget,
                                                     int32_t group_size) {
  const int32_t kv_head = blockIdx.x;
  const int32_t batch_idx = blockIdx.y;
  const int32_t tid = threadIdx.x;
  if (batch_idx >= batch_size || kv_head >= num_kv_heads) {
    return;
  }

  const int32_t total_selected = max_group_selected + exception_budget;
  extern __shared__ float smem[];
  float* logits = smem;
  float* scratch = logits + group_size * total_selected;

  const float scale = rsqrtf(static_cast<float>(head_dim));
  const int32_t head_begin = kv_head * group_size;
  const int64_t q_batch_base = static_cast<int64_t>(batch_idx) * num_heads * head_dim;
  const int64_t kv_batch_base =
      static_cast<int64_t>(batch_idx) * num_kv_heads * seq_len * head_dim;
  const int64_t group_index_base =
      (static_cast<int64_t>(batch_idx) * num_kv_heads + kv_head) * max_group_selected;
  const int64_t out_batch_base = q_batch_base;

  for (int32_t selected_pos = tid; selected_pos < max_group_selected;
       selected_pos += blockDim.x) {
    const int32_t token = group_indices[group_index_base + selected_pos];
    float dots[kMaxGroupSize];
#pragma unroll
    for (int32_t h = 0; h < kMaxGroupSize; ++h) {
      dots[h] = 0.0f;
    }
    if (token >= 0 && token < seq_len) {
      const int64_t k_base =
          kv_batch_base + (static_cast<int64_t>(kv_head) * seq_len + token) * head_dim;
      for (int32_t dim = 0; dim < head_dim; ++dim) {
        const float k_val = __half2float(k[k_base + dim]);
        for (int32_t h = 0; h < group_size; ++h) {
          const int32_t q_head = head_begin + h;
          dots[h] += __half2float(q[q_batch_base + q_head * head_dim + dim]) * k_val;
        }
      }
    }
    for (int32_t h = 0; h < group_size; ++h) {
      logits[h * total_selected + selected_pos] =
          (token >= 0 && token < seq_len) ? dots[h] * scale : -FLT_MAX;
    }
  }

  const int32_t exception_logits_count = group_size * exception_budget;
  for (int32_t linear = tid; linear < exception_logits_count; linear += blockDim.x) {
    const int32_t h = linear / exception_budget;
    const int32_t exception_pos = linear - h * exception_budget;
    const int32_t q_head = head_begin + h;
    const int64_t exception_index_base =
        (static_cast<int64_t>(batch_idx) * num_heads + q_head) * exception_budget;
    const int32_t token = exception_indices[exception_index_base + exception_pos];
    float dot = 0.0f;
    if (token >= 0 && token < seq_len) {
      const int64_t k_base =
          kv_batch_base + (static_cast<int64_t>(kv_head) * seq_len + token) * head_dim;
      const int64_t q_base = q_batch_base + q_head * head_dim;
      for (int32_t dim = 0; dim < head_dim; ++dim) {
        dot += __half2float(q[q_base + dim]) * __half2float(k[k_base + dim]);
      }
    }
    logits[h * total_selected + max_group_selected + exception_pos] =
        (token >= 0 && token < seq_len) ? dot * scale : -FLT_MAX;
  }
  __syncthreads();

  for (int32_t h = 0; h < group_size; ++h) {
    float local_max = -FLT_MAX;
    for (int32_t selected_pos = tid; selected_pos < total_selected; selected_pos += blockDim.x) {
      local_max = fmaxf(local_max, logits[h * total_selected + selected_pos]);
    }
    scratch[tid] = local_max;
    __syncthreads();
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
      }
      __syncthreads();
    }
    const float max_val = scratch[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int32_t selected_pos = tid; selected_pos < total_selected; selected_pos += blockDim.x) {
      const int32_t offset = h * total_selected + selected_pos;
      const float exp_val = expf(logits[offset] - max_val);
      logits[offset] = exp_val;
      local_sum += exp_val;
    }
    scratch[tid] = local_sum;
    __syncthreads();
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        scratch[tid] += scratch[tid + stride];
      }
      __syncthreads();
    }
    const float inv_sum = 1.0f / scratch[0];

    for (int32_t selected_pos = tid; selected_pos < total_selected; selected_pos += blockDim.x) {
      logits[h * total_selected + selected_pos] *= inv_sum;
    }
    __syncthreads();
  }

  for (int32_t dim = tid; dim < head_dim; dim += blockDim.x) {
    float acc[kMaxGroupSize];
#pragma unroll
    for (int32_t h = 0; h < kMaxGroupSize; ++h) {
      acc[h] = 0.0f;
    }
    for (int32_t selected_pos = 0; selected_pos < max_group_selected; ++selected_pos) {
      const int32_t token = group_indices[group_index_base + selected_pos];
      if (token < 0 || token >= seq_len) {
        continue;
      }
      const int64_t v_base =
          kv_batch_base + (static_cast<int64_t>(kv_head) * seq_len + token) * head_dim;
      const float v_val = __half2float(v[v_base + dim]);
      for (int32_t h = 0; h < group_size; ++h) {
        acc[h] += logits[h * total_selected + selected_pos] * v_val;
      }
    }
    for (int32_t h = 0; h < group_size; ++h) {
      const int32_t q_head = head_begin + h;
      const int64_t exception_index_base =
          (static_cast<int64_t>(batch_idx) * num_heads + q_head) * exception_budget;
      for (int32_t exception_pos = 0; exception_pos < exception_budget; ++exception_pos) {
        const int32_t token = exception_indices[exception_index_base + exception_pos];
        if (token < 0 || token >= seq_len) {
          continue;
        }
        const int64_t v_base =
            kv_batch_base + (static_cast<int64_t>(kv_head) * seq_len + token) * head_dim;
        acc[h] += logits[h * total_selected + max_group_selected + exception_pos] *
                  __half2float(v[v_base + dim]);
      }
      out[out_batch_base + q_head * head_dim + dim] = __float2half(acc[h]);
    }
  }
}

__global__ void BatchedPerHeadSparseDecodeKernel(const half* __restrict__ q,
                                                 const half* __restrict__ k,
                                                 const half* __restrict__ v,
                                                 const int32_t* __restrict__ indices,
                                                 half* __restrict__ out,
                                                 int32_t batch_size,
                                                 int32_t num_heads,
                                                 int32_t num_kv_heads,
                                                 int32_t seq_len,
                                                 int32_t head_dim,
                                                 int32_t max_selected,
                                                 int32_t group_size) {
  const int32_t q_head = blockIdx.x;
  const int32_t batch_idx = blockIdx.y;
  const int32_t tid = threadIdx.x;
  if (batch_idx >= batch_size || q_head >= num_heads) {
    return;
  }

  extern __shared__ float smem[];
  float* logits = smem;
  float* scratch = logits + max_selected;

  const int32_t kv_head = q_head / group_size;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  const int64_t q_batch_base = static_cast<int64_t>(batch_idx) * num_heads * head_dim;
  const int64_t kv_batch_base =
      static_cast<int64_t>(batch_idx) * num_kv_heads * seq_len * head_dim;
  const int64_t index_base =
      (static_cast<int64_t>(batch_idx) * num_heads + q_head) * max_selected;

  for (int32_t selected_pos = tid; selected_pos < max_selected; selected_pos += blockDim.x) {
    const int32_t token = indices[index_base + selected_pos];
    float dot = 0.0f;
    if (token >= 0 && token < seq_len) {
      const int64_t k_base =
          kv_batch_base + (static_cast<int64_t>(kv_head) * seq_len + token) * head_dim;
      const int64_t q_base = q_batch_base + q_head * head_dim;
      for (int32_t dim = 0; dim < head_dim; ++dim) {
        dot += __half2float(q[q_base + dim]) * __half2float(k[k_base + dim]);
      }
    }
    logits[selected_pos] = (token >= 0 && token < seq_len) ? dot * scale : -FLT_MAX;
  }
  __syncthreads();

  float local_max = -FLT_MAX;
  for (int32_t selected_pos = tid; selected_pos < max_selected; selected_pos += blockDim.x) {
    local_max = fmaxf(local_max, logits[selected_pos]);
  }
  scratch[tid] = local_max;
  __syncthreads();
  for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
    }
    __syncthreads();
  }
  const float max_val = scratch[0];
  __syncthreads();

  float local_sum = 0.0f;
  for (int32_t selected_pos = tid; selected_pos < max_selected; selected_pos += blockDim.x) {
    const float exp_val = expf(logits[selected_pos] - max_val);
    logits[selected_pos] = exp_val;
    local_sum += exp_val;
  }
  scratch[tid] = local_sum;
  __syncthreads();
  for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  const float inv_sum = 1.0f / scratch[0];
  for (int32_t selected_pos = tid; selected_pos < max_selected; selected_pos += blockDim.x) {
    logits[selected_pos] *= inv_sum;
  }
  __syncthreads();

  for (int32_t dim = tid; dim < head_dim; dim += blockDim.x) {
    float acc = 0.0f;
    for (int32_t selected_pos = 0; selected_pos < max_selected; ++selected_pos) {
      const int32_t token = indices[index_base + selected_pos];
      if (token < 0 || token >= seq_len) {
        continue;
      }
      const int64_t v_base =
          kv_batch_base + (static_cast<int64_t>(kv_head) * seq_len + token) * head_dim;
      acc += logits[selected_pos] * __half2float(v[v_base + dim]);
    }
    out[q_batch_base + q_head * head_dim + dim] = __float2half(acc);
  }
}

void CheckInput(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void CheckCommonInput(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor indices) {
  CheckInput(q, "q");
  CheckInput(k, "k");
  CheckInput(v, "v");
  CheckInput(indices, "indices");
  TORCH_CHECK(q.scalar_type() == at::kHalf, "q must be float16");
  TORCH_CHECK(k.scalar_type() == at::kHalf, "k must be float16");
  TORCH_CHECK(v.scalar_type() == at::kHalf, "v must be float16");
  TORCH_CHECK(indices.scalar_type() == at::kInt, "indices must be int32");
}

void SetSharedMemory(size_t dynamic_smem) {
  TORCH_CHECK(dynamic_smem <= 96 * 1024,
              "dynamic shared memory requirement is too large: ", dynamic_smem, " bytes");
  cudaFuncSetAttribute(BatchedGQAGroupSharedDecodeKernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       static_cast<int>(dynamic_smem));
  cudaFuncSetAttribute(BatchedGQAGroupExceptionDecodeKernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       static_cast<int>(dynamic_smem));
  cudaFuncSetAttribute(BatchedPerHeadSparseDecodeKernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       static_cast<int>(dynamic_smem));
}

}  // namespace

at::Tensor batched_gqa_group_shared_decode(at::Tensor q,
                                           at::Tensor k,
                                           at::Tensor v,
                                           at::Tensor indices) {
  CheckCommonInput(q, k, v, indices);
  TORCH_CHECK(q.dim() == 3, "q must have shape [batch, num_heads, head_dim]");
  TORCH_CHECK(k.dim() == 4, "k must have shape [batch, num_kv_heads, seq_len, head_dim]");
  TORCH_CHECK(v.sizes() == k.sizes(), "v must have the same shape as k");
  TORCH_CHECK(indices.dim() == 3,
              "indices must have shape [batch, num_kv_heads, max_selected]");

  const int32_t batch_size = static_cast<int32_t>(q.size(0));
  const int32_t num_heads = static_cast<int32_t>(q.size(1));
  const int32_t head_dim = static_cast<int32_t>(q.size(2));
  const int32_t num_kv_heads = static_cast<int32_t>(k.size(1));
  const int32_t seq_len = static_cast<int32_t>(k.size(2));
  const int32_t max_selected = static_cast<int32_t>(indices.size(2));
  TORCH_CHECK(k.size(0) == batch_size, "k batch must match q batch");
  TORCH_CHECK(k.size(3) == head_dim, "k head_dim must match q head_dim");
  TORCH_CHECK(indices.size(0) == batch_size, "indices batch must match q batch");
  TORCH_CHECK(indices.size(1) == num_kv_heads, "indices num_kv_heads must match k");
  TORCH_CHECK(num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads");
  const int32_t group_size = num_heads / num_kv_heads;
  TORCH_CHECK(group_size > 0 && group_size <= kMaxGroupSize,
              "group_size must be in [1, ", kMaxGroupSize, "]");
  TORCH_CHECK(max_selected > 0, "max_selected must be positive");

  const size_t dynamic_smem =
      static_cast<size_t>(group_size) * max_selected * sizeof(float) +
      kBlockThreads * sizeof(float);
  SetSharedMemory(dynamic_smem);

  auto out = at::empty({batch_size, num_heads, head_dim}, q.options());
  BatchedGQAGroupSharedDecodeKernel<<<dim3(num_kv_heads, batch_size), kBlockThreads,
                                      dynamic_smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
      indices.data_ptr<int32_t>(),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      batch_size,
      num_heads,
      num_kv_heads,
      seq_len,
      head_dim,
      max_selected,
      group_size);
  const cudaError_t status = cudaGetLastError();
  TORCH_CHECK(status == cudaSuccess,
              "BatchedGQAGroupSharedDecodeKernel failed: ", cudaGetErrorString(status));
  return out;
}

std::vector<at::Tensor> build_group_max_exception_indices(at::Tensor estimated_weights,
                                                          at::Tensor candidate_mask,
                                                          int64_t num_key_value_heads,
                                                          double threshold,
                                                          int64_t max_group_selected,
                                                          int64_t exception_budget) {
  CheckInput(estimated_weights, "estimated_weights");
  CheckInput(candidate_mask, "candidate_mask");
  TORCH_CHECK(estimated_weights.scalar_type() == at::kFloat,
              "estimated_weights must be float32");
  TORCH_CHECK(candidate_mask.scalar_type() == at::kBool,
              "candidate_mask must be bool");
  TORCH_CHECK(estimated_weights.dim() == 4,
              "estimated_weights must have shape [batch, num_heads, 1, seq_len]");
  TORCH_CHECK(candidate_mask.sizes() == estimated_weights.sizes(),
              "candidate_mask must have the same shape as estimated_weights");
  TORCH_CHECK(estimated_weights.size(2) == 1,
              "estimated_weights q_len dimension must be 1");
  const int32_t batch_size = static_cast<int32_t>(estimated_weights.size(0));
  const int32_t num_heads = static_cast<int32_t>(estimated_weights.size(1));
  const int32_t seq_len = static_cast<int32_t>(estimated_weights.size(3));
  const int32_t num_kv_heads = static_cast<int32_t>(num_key_value_heads);
  const int32_t max_group = static_cast<int32_t>(max_group_selected);
  const int32_t exception = static_cast<int32_t>(exception_budget);

  TORCH_CHECK(num_kv_heads > 0, "num_key_value_heads must be positive");
  TORCH_CHECK(num_heads % num_kv_heads == 0,
              "num_heads must be divisible by num_key_value_heads");
  const int32_t group_size = num_heads / num_kv_heads;
  TORCH_CHECK(group_size > 0 && group_size <= kMaxGroupSize,
              "group_size must be in [1, ", kMaxGroupSize, "]");
  TORCH_CHECK(max_group > 0, "max_group_selected must be positive");
  TORCH_CHECK(exception >= 0, "exception_budget must be non-negative");

  auto int_options = estimated_weights.options().dtype(at::kInt);
  auto group_indices = at::empty({batch_size, num_kv_heads, max_group}, int_options);
  auto group_counts = at::empty({batch_size, num_kv_heads}, int_options);
  auto exception_indices = at::empty({batch_size, num_heads, exception}, int_options);
  auto exception_counts = at::empty({batch_size, num_heads}, int_options);

  const size_t group_smem =
      static_cast<size_t>(max_group) * sizeof(float) +
      static_cast<size_t>(max_group) * sizeof(int32_t) +
      kBlockThreads * sizeof(float);
  TORCH_CHECK(group_smem <= 96 * 1024,
              "group compaction shared memory requirement is too large: ",
              group_smem, " bytes");
  cudaFuncSetAttribute(BuildGroupMaxSharedIndicesKernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       static_cast<int>(group_smem));

  BuildGroupMaxSharedIndicesKernel<<<dim3(num_kv_heads, batch_size), kBlockThreads,
                                     group_smem, at::cuda::getCurrentCUDAStream()>>>(
      estimated_weights.data_ptr<float>(),
      candidate_mask.data_ptr<bool>(),
      group_indices.data_ptr<int32_t>(),
      group_counts.data_ptr<int32_t>(),
      batch_size,
      num_heads,
      num_kv_heads,
      seq_len,
      max_group,
      static_cast<float>(threshold),
      group_size);
  cudaError_t status = cudaGetLastError();
  TORCH_CHECK(status == cudaSuccess,
              "BuildGroupMaxSharedIndicesKernel failed: ", cudaGetErrorString(status));

  BuildPerHeadExceptionIndicesKernel<<<dim3(num_heads, batch_size), kBlockThreads, 0,
                                       at::cuda::getCurrentCUDAStream()>>>(
      estimated_weights.data_ptr<float>(),
      candidate_mask.data_ptr<bool>(),
      group_indices.data_ptr<int32_t>(),
      exception_indices.data_ptr<int32_t>(),
      exception_counts.data_ptr<int32_t>(),
      batch_size,
      num_heads,
      num_kv_heads,
      seq_len,
      max_group,
      exception,
      group_size);
  status = cudaGetLastError();
  TORCH_CHECK(status == cudaSuccess,
              "BuildPerHeadExceptionIndicesKernel failed: ", cudaGetErrorString(status));

  return {group_indices, exception_indices, group_counts, exception_counts};
}

std::vector<at::Tensor> compact_group_exception_masks(at::Tensor shared_mask,
                                                      at::Tensor final_mask,
                                                      int64_t num_key_value_heads,
                                                      int64_t max_group_selected,
                                                      int64_t exception_budget) {
  CheckInput(shared_mask, "shared_mask");
  CheckInput(final_mask, "final_mask");
  TORCH_CHECK(shared_mask.scalar_type() == at::kBool, "shared_mask must be bool");
  TORCH_CHECK(final_mask.scalar_type() == at::kBool, "final_mask must be bool");
  TORCH_CHECK(shared_mask.dim() == 4,
              "shared_mask must have shape [batch, num_heads, 1, seq_len]");
  TORCH_CHECK(final_mask.sizes() == shared_mask.sizes(),
              "final_mask must have the same shape as shared_mask");
  TORCH_CHECK(shared_mask.size(2) == 1, "shared_mask q_len dimension must be 1");

  const int32_t batch_size = static_cast<int32_t>(shared_mask.size(0));
  const int32_t num_heads = static_cast<int32_t>(shared_mask.size(1));
  const int32_t seq_len = static_cast<int32_t>(shared_mask.size(3));
  const int32_t num_kv_heads = static_cast<int32_t>(num_key_value_heads);
  const int32_t max_group = static_cast<int32_t>(max_group_selected);
  const int32_t exception = static_cast<int32_t>(exception_budget);
  TORCH_CHECK(num_kv_heads > 0, "num_key_value_heads must be positive");
  TORCH_CHECK(num_heads % num_kv_heads == 0,
              "num_heads must be divisible by num_key_value_heads");
  const int32_t group_size = num_heads / num_kv_heads;
  TORCH_CHECK(group_size > 0 && group_size <= kMaxGroupSize,
              "group_size must be in [1, ", kMaxGroupSize, "]");
  TORCH_CHECK(max_group > 0, "max_group_selected must be positive");
  TORCH_CHECK(exception >= 0, "exception_budget must be non-negative");

  auto int_options = shared_mask.options().dtype(at::kInt);
  auto group_indices = at::empty({batch_size, num_kv_heads, max_group}, int_options);
  auto group_counts = at::empty({batch_size, num_kv_heads}, int_options);
  auto exception_indices = at::empty({batch_size, num_heads, exception}, int_options);
  auto exception_counts = at::empty({batch_size, num_heads}, int_options);

  CompactGroupMaskKernel<<<dim3(num_kv_heads, batch_size), kBlockThreads, 0,
                           at::cuda::getCurrentCUDAStream()>>>(
      shared_mask.data_ptr<bool>(),
      group_indices.data_ptr<int32_t>(),
      group_counts.data_ptr<int32_t>(),
      batch_size,
      num_heads,
      num_kv_heads,
      seq_len,
      max_group,
      group_size);
  cudaError_t status = cudaGetLastError();
  TORCH_CHECK(status == cudaSuccess,
              "CompactGroupMaskKernel failed: ", cudaGetErrorString(status));

  CompactExceptionMaskKernel<<<dim3(num_heads, batch_size), kBlockThreads, 0,
                               at::cuda::getCurrentCUDAStream()>>>(
      final_mask.data_ptr<bool>(),
      shared_mask.data_ptr<bool>(),
      exception_indices.data_ptr<int32_t>(),
      exception_counts.data_ptr<int32_t>(),
      batch_size,
      num_heads,
      seq_len,
      exception);
  status = cudaGetLastError();
  TORCH_CHECK(status == cudaSuccess,
              "CompactExceptionMaskKernel failed: ", cudaGetErrorString(status));

  return {group_indices, exception_indices, group_counts, exception_counts};
}

at::Tensor batched_per_head_sparse_decode_out(at::Tensor q,
                                              at::Tensor k,
                                              at::Tensor v,
                                              at::Tensor indices,
                                              at::Tensor out) {
  CheckCommonInput(q, k, v, indices);
  CheckInput(out, "out");
  TORCH_CHECK(q.dim() == 3, "q must have shape [batch, num_heads, head_dim]");
  TORCH_CHECK(k.dim() == 4, "k must have shape [batch, num_kv_heads, seq_len, head_dim]");
  TORCH_CHECK(v.sizes() == k.sizes(), "v must have the same shape as k");
  TORCH_CHECK(indices.dim() == 3,
              "indices must have shape [batch, num_heads, max_selected]");

  const int32_t batch_size = static_cast<int32_t>(q.size(0));
  const int32_t num_heads = static_cast<int32_t>(q.size(1));
  const int32_t head_dim = static_cast<int32_t>(q.size(2));
  const int32_t num_kv_heads = static_cast<int32_t>(k.size(1));
  const int32_t seq_len = static_cast<int32_t>(k.size(2));
  const int32_t max_selected = static_cast<int32_t>(indices.size(2));
  TORCH_CHECK(k.size(0) == batch_size, "k batch must match q batch");
  TORCH_CHECK(k.size(3) == head_dim, "k head_dim must match q head_dim");
  TORCH_CHECK(indices.size(0) == batch_size, "indices batch must match q batch");
  TORCH_CHECK(indices.size(1) == num_heads, "indices num_heads must match q");
  TORCH_CHECK(out.scalar_type() == at::kHalf, "out must be float16");
  TORCH_CHECK(out.sizes() == q.sizes(), "out must have the same shape as q");
  TORCH_CHECK(out.get_device() == q.get_device(), "out must be on the same device as q");
  TORCH_CHECK(num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads");
  const int32_t group_size = num_heads / num_kv_heads;
  TORCH_CHECK(group_size > 0 && group_size <= kMaxGroupSize,
              "group_size must be in [1, ", kMaxGroupSize, "]");
  TORCH_CHECK(max_selected > 0, "max_selected must be positive");

  const size_t dynamic_smem =
      static_cast<size_t>(max_selected) * sizeof(float) + kBlockThreads * sizeof(float);
  SetSharedMemory(dynamic_smem);

  BatchedPerHeadSparseDecodeKernel<<<dim3(num_heads, batch_size), kBlockThreads,
                                     dynamic_smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
      indices.data_ptr<int32_t>(),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      batch_size,
      num_heads,
      num_kv_heads,
      seq_len,
      head_dim,
      max_selected,
      group_size);
  const cudaError_t status = cudaGetLastError();
  TORCH_CHECK(status == cudaSuccess,
              "BatchedPerHeadSparseDecodeKernel failed: ", cudaGetErrorString(status));
  return out;
}

at::Tensor batched_per_head_sparse_decode(at::Tensor q,
                                          at::Tensor k,
                                          at::Tensor v,
                                          at::Tensor indices) {
  TORCH_CHECK(q.dim() == 3, "q must have shape [batch, num_heads, head_dim]");
  auto out = at::empty_like(q);
  return batched_per_head_sparse_decode_out(q, k, v, indices, out);
}

at::Tensor batched_gqa_group_exception_decode_out(at::Tensor q,
                                                  at::Tensor k,
                                                  at::Tensor v,
                                                  at::Tensor group_indices,
                                                  at::Tensor exception_indices,
                                                  at::Tensor out) {
  CheckCommonInput(q, k, v, group_indices);
  CheckInput(exception_indices, "exception_indices");
  CheckInput(out, "out");
  TORCH_CHECK(exception_indices.scalar_type() == at::kInt,
              "exception_indices must be int32");
  TORCH_CHECK(q.dim() == 3, "q must have shape [batch, num_heads, head_dim]");
  TORCH_CHECK(k.dim() == 4, "k must have shape [batch, num_kv_heads, seq_len, head_dim]");
  TORCH_CHECK(v.sizes() == k.sizes(), "v must have the same shape as k");
  TORCH_CHECK(group_indices.dim() == 3,
              "group_indices must have shape [batch, num_kv_heads, max_group_selected]");
  TORCH_CHECK(exception_indices.dim() == 3,
              "exception_indices must have shape [batch, num_heads, exception_budget]");

  const int32_t batch_size = static_cast<int32_t>(q.size(0));
  const int32_t num_heads = static_cast<int32_t>(q.size(1));
  const int32_t head_dim = static_cast<int32_t>(q.size(2));
  const int32_t num_kv_heads = static_cast<int32_t>(k.size(1));
  const int32_t seq_len = static_cast<int32_t>(k.size(2));
  const int32_t max_group_selected = static_cast<int32_t>(group_indices.size(2));
  const int32_t exception_budget = static_cast<int32_t>(exception_indices.size(2));
  TORCH_CHECK(k.size(0) == batch_size, "k batch must match q batch");
  TORCH_CHECK(k.size(3) == head_dim, "k head_dim must match q head_dim");
  TORCH_CHECK(group_indices.size(0) == batch_size, "group_indices batch must match q");
  TORCH_CHECK(group_indices.size(1) == num_kv_heads,
              "group_indices num_kv_heads must match k");
  TORCH_CHECK(exception_indices.size(0) == batch_size,
              "exception_indices batch must match q");
  TORCH_CHECK(exception_indices.size(1) == num_heads,
              "exception_indices num_heads must match q");
  TORCH_CHECK(out.scalar_type() == at::kHalf, "out must be float16");
  TORCH_CHECK(out.sizes() == q.sizes(), "out must have the same shape as q");
  TORCH_CHECK(out.get_device() == q.get_device(), "out must be on the same device as q");
  TORCH_CHECK(num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads");
  const int32_t group_size = num_heads / num_kv_heads;
  TORCH_CHECK(group_size > 0 && group_size <= kMaxGroupSize,
              "group_size must be in [1, ", kMaxGroupSize, "]");
  TORCH_CHECK(max_group_selected > 0, "max_group_selected must be positive");
  TORCH_CHECK(exception_budget >= 0, "exception_budget must be non-negative");

  const int32_t total_selected = max_group_selected + exception_budget;
  const size_t dynamic_smem =
      static_cast<size_t>(group_size) * total_selected * sizeof(float) +
      kBlockThreads * sizeof(float);
  SetSharedMemory(dynamic_smem);

  BatchedGQAGroupExceptionDecodeKernel<<<dim3(num_kv_heads, batch_size), kBlockThreads,
                                         dynamic_smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
      group_indices.data_ptr<int32_t>(),
      exception_indices.data_ptr<int32_t>(),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      batch_size,
      num_heads,
      num_kv_heads,
      seq_len,
      head_dim,
      max_group_selected,
      exception_budget,
      group_size);
  const cudaError_t status = cudaGetLastError();
  TORCH_CHECK(status == cudaSuccess,
              "BatchedGQAGroupExceptionDecodeKernel failed: ", cudaGetErrorString(status));
  return out;
}

at::Tensor batched_gqa_group_exception_decode(at::Tensor q,
                                              at::Tensor k,
                                              at::Tensor v,
                                              at::Tensor group_indices,
                                              at::Tensor exception_indices) {
  TORCH_CHECK(q.dim() == 3, "q must have shape [batch, num_heads, head_dim]");
  auto out = at::empty_like(q);
  return batched_gqa_group_exception_decode_out(
      q, k, v, group_indices, exception_indices, out);
}

at::Tensor gqa_group_shared_decode(at::Tensor q,
                                   at::Tensor k,
                                   at::Tensor v,
                                   at::Tensor indices) {
  CheckInput(q, "q");
  CheckInput(k, "k");
  CheckInput(v, "v");
  CheckInput(indices, "indices");
  TORCH_CHECK(q.scalar_type() == at::kHalf, "q must be float16");
  TORCH_CHECK(k.scalar_type() == at::kHalf, "k must be float16");
  TORCH_CHECK(v.scalar_type() == at::kHalf, "v must be float16");
  TORCH_CHECK(indices.scalar_type() == at::kInt, "indices must be int32");
  TORCH_CHECK(q.dim() == 2, "q must have shape [num_heads, head_dim]");
  TORCH_CHECK(k.dim() == 3, "k must have shape [num_kv_heads, seq_len, head_dim]");
  TORCH_CHECK(v.sizes() == k.sizes(), "v must have the same shape as k");
  TORCH_CHECK(indices.dim() == 2, "indices must have shape [num_kv_heads, max_selected]");

  const int32_t num_heads = static_cast<int32_t>(q.size(0));
  const int32_t head_dim = static_cast<int32_t>(q.size(1));
  const int32_t num_kv_heads = static_cast<int32_t>(k.size(0));
  const int32_t seq_len = static_cast<int32_t>(k.size(1));
  const int32_t max_selected = static_cast<int32_t>(indices.size(1));
  TORCH_CHECK(k.size(2) == head_dim, "k head_dim must match q head_dim");
  TORCH_CHECK(indices.size(0) == num_kv_heads, "indices num_kv_heads must match k");
  TORCH_CHECK(num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads");
  const int32_t group_size = num_heads / num_kv_heads;
  TORCH_CHECK(group_size > 0 && group_size <= kMaxGroupSize,
              "group_size must be in [1, ", kMaxGroupSize, "]");
  TORCH_CHECK(max_selected > 0, "max_selected must be positive");

  const size_t dynamic_smem =
      static_cast<size_t>(group_size) * max_selected * sizeof(float) +
      kBlockThreads * sizeof(float);
  SetSharedMemory(dynamic_smem);

  auto out = at::empty({num_heads, head_dim}, q.options());
  cudaFuncSetAttribute(GQAGroupSharedDecodeKernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       static_cast<int>(dynamic_smem));
  GQAGroupSharedDecodeKernel<<<num_kv_heads, kBlockThreads, dynamic_smem,
                               at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
      indices.data_ptr<int32_t>(),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      num_heads,
      num_kv_heads,
      seq_len,
      head_dim,
      max_selected,
      group_size);
  const cudaError_t status = cudaGetLastError();
  TORCH_CHECK(status == cudaSuccess,
              "GQAGroupSharedDecodeKernel failed: ", cudaGetErrorString(status));
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gqa_group_shared_decode", &gqa_group_shared_decode,
        "Fused group-shared sparse decode attention for GQA");
  m.def("batched_gqa_group_shared_decode", &batched_gqa_group_shared_decode,
        "Batched fused group-shared sparse decode attention for GQA");
  m.def("batched_gqa_group_exception_decode", &batched_gqa_group_exception_decode,
        "Batched fused group-shared sparse decode attention with per-head exceptions");
  m.def("batched_gqa_group_exception_decode_out", &batched_gqa_group_exception_decode_out,
        "Batched fused group-shared sparse decode attention with preallocated output");
  m.def("build_group_max_exception_indices", &build_group_max_exception_indices,
        "Build compact group_max shared and per-head exception indices");
  m.def("compact_group_exception_masks", &compact_group_exception_masks,
        "Compact precomputed group-shared and final masks into decode indices");
  m.def("batched_per_head_sparse_decode", &batched_per_head_sparse_decode,
        "Batched fused per-head sparse decode attention");
  m.def("batched_per_head_sparse_decode_out", &batched_per_head_sparse_decode_out,
        "Batched fused per-head sparse decode attention with preallocated output");
}
