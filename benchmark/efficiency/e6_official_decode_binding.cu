// Minimal pybind interface to the unmodified upstream approx_attn.cu object.
// The interface follows FlashInfer-derived upstream code, Apache-2.0.
#include <torch/extension.h>
#include "decode/decode_handler.cuh"

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
  void Forward(torch::Tensor q, torch::Tensor o, torch::Tensor paged_kv_data,
               torch::Tensor paged_kv_indices, torch::Tensor paged_kv_indptr,
               unsigned int paged_kv_last_page_len, unsigned int paged_kv_last_page_idx,
               torch::Tensor qk_product, float rope_scale, float rope_theta);
#else
  void Forward(torch::Tensor q, torch::Tensor o, torch::Tensor paged_kv_data,
               torch::Tensor paged_kv_indices, torch::Tensor paged_kv_indptr,
               unsigned int paged_kv_last_page_len, unsigned int paged_kv_last_page_idx,
               float rope_scale, float rope_theta);
#endif
 private:
  explicit BatchDecodeWithPagedKVCachePyTorchWrapper(unsigned int layout)
      : kv_layout_(flashinfer::QKVLayout(layout)) {}
  flashinfer::BatchDecodeHandler handler_;
  flashinfer::QKVLayout kv_layout_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<BatchDecodeWithPagedKVCachePyTorchWrapper>(
      m, "BatchDecodeWithPagedKVCachePyTorchWrapper")
      .def(pybind11::init(&BatchDecodeWithPagedKVCachePyTorchWrapper::Create))
      .def("begin_forward", &BatchDecodeWithPagedKVCachePyTorchWrapper::BeginForward)
      .def("end_forward", &BatchDecodeWithPagedKVCachePyTorchWrapper::EndForward)
      .def("forward", &BatchDecodeWithPagedKVCachePyTorchWrapper::Forward);
}
