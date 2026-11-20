#pragma once
#include <torch/extension.h>
#include "decode/decode_handler.cuh"

// Matches the author wrapper declaration and the standalone pybind bridge.
class BatchDecodeWithPagedKVCachePyTorchWrapper {
 public:
  static BatchDecodeWithPagedKVCachePyTorchWrapper Create(unsigned int layout) {
    return BatchDecodeWithPagedKVCachePyTorchWrapper(layout);
  }
  void BeginForward(torch::Tensor indptr, unsigned int num_qo_heads,
                    unsigned int num_kv_heads, unsigned int head_dim,
                    unsigned int page_size, torch::Tensor empty_data);
  void EndForward();
#ifdef E6_LESSISMORE
  void Forward(torch::Tensor q, torch::Tensor o, torch::Tensor data, torch::Tensor indices,
               torch::Tensor indptr, unsigned int last_len, unsigned int last_idx,
               torch::Tensor qk_product, float rope_scale, float rope_theta);
#else
  void Forward(torch::Tensor q, torch::Tensor o, torch::Tensor data, torch::Tensor indices,
               torch::Tensor indptr, unsigned int last_len, unsigned int last_idx,
               float rope_scale, float rope_theta);
#endif
 private:
  explicit BatchDecodeWithPagedKVCachePyTorchWrapper(unsigned int layout)
      : kv_layout_(flashinfer::QKVLayout(layout)) {}
  flashinfer::BatchDecodeHandler handler_;
  flashinfer::QKVLayout kv_layout_;
};
