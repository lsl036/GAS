// Implementations of fast dequantization algorithms.
//
// Learned from:
//  https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h

#ifndef FTKA_DEQUANTIZE_CUH_
#define FTKA_DEQUANTIZE_CUH_

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "flashinfer/vec_dtypes.cuh"
#include "lowbit_dtypes.cuh"

using namespace flashinfer;

namespace ftka {
namespace dequant {

// This converter is meant to be used with data interleaved in a 32-bit register where the even
// elements are in the low bits and the odd elemeents are in the high bits of the register. In
// addition, it assumes elements were originally signed and had a bias of 2**(b-1) added (where b is
// the number of bits in the type) to make all numbers unsigned. This converter will uninterleave
// the data and subtract the bias while converting to the result type.

/*!
 * \brief Dequantize the input (a vec of int8_t) into vec_t<float>
 * \tparam vec_size A template integer indicates the vector size
 * \param src Const input data
 * \param tgt Output data
 * \param scale Quantization parameter
 * \param zero_point Quantization parameter
 */
template <size_t vec_size, typename src_type>
FLASHINFER_INLINE void fast_dequantize_i8s_to_fp16s(const src_type* src, vec_t<half, vec_size>& tgt,
                                                    half scale, half zero) {
  // #pragma unroll
  //   for (int i = 0; i < vec_size; i++) {
  //       *reinterpret_cast<uint8_t*>(tgt.ptr() + i) = src[i];
  //   }
  //   return;

  static constexpr int VEC_WIDTH = 4;
  if (!std::is_same_v<src_type, uint8_t>) {
    // NOTE(chaofan): to pass the compilation
    return;
  }
  static_assert(vec_size % VEC_WIDTH == 0, "Invalid vector size");

#pragma unroll
  for (int i = 0; i < vec_size; i += VEC_WIDTH) {
    uint32_t h[2];  // 64bits = 4 * fp16
    uint32_t const i8s = *reinterpret_cast<uint32_t const*>(src + i);

    static constexpr uint32_t mask_for_elt_01 = 0x5250;
    static constexpr uint32_t mask_for_elt_23 = 0x5351;
    static constexpr uint32_t start_byte_for_fp16 = 0x64646464;
    asm volatile("prmt.b32 %0,%1,%2,%3;\n"
                 : "=r"(h[0])
                 : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
    asm volatile("prmt.b32 %0,%1,%2,%3;\n"
                 : "=r"(h[1])
                 : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));

    // Lastly, we subtract 1152 from our constructed number using fp16 math to get our signed
    // integer as fp16.
    static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(I8s_TO_F16s_MAGIC_NUM));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[1]) : "r"(h[1]), "r"(I8s_TO_F16s_MAGIC_NUM));

    // Scale & Zero
    half2 scale2 = make_half2(scale, scale);
    half2 zero2 = make_half2(zero, zero);
    half2* result = reinterpret_cast<half2*>(h);

    // asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[0]) : "r"(scale2), "r"(zero2),
    // "r"(h[0])); asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[1]) : "r"(scale2),
    // "r"(zero2), "r"(h[1]));

    result[0] = __hfma2(result[0], scale2, zero2);
    result[1] = __hfma2(result[1], scale2, zero2);

    uint32_t* tgt_u32 = reinterpret_cast<uint32_t*>(tgt.ptr() + i);
    tgt_u32[0] = h[0];
    tgt_u32[1] = h[1];

    // half* h_half = reinterpret_cast<half*>(h);

    // half2* h_half2 = reinterpret_cast<half2*>(h);
    // float2* tgt_ptr = reinterpret_cast<float2*>(tgt.ptr() + i);
    // tgt_ptr[0] = __half22float2(h_half2[0]);
    // tgt_ptr[1] = __half22float2(h_half2[1]);
  }
}

/*!
 * \brief Dequantize the input (a vec of int4_t) into vec_t<float>
 * \tparam vec_size A template integer indicates the vector size
 * \param src Const input data
 * \param tgt Output data
 * \param scale Quantization parameter
 * \param zero_point Quantization parameter
 */
template <size_t vec_size, typename src_type>
FLASHINFER_INLINE void fast_dequantize_i4s_to_fp16s(const src_type* src, vec_t<half, vec_size>& tgt,
                                                    half scale, half zero) {
  if (!std::is_same_v<src_type, lowbit::uint_4bit>) {
    // NOTE(chaofan): to pass the compilation
    return;
  }
  const uint8_t* src_u8 = reinterpret_cast<const uint8_t*>(src);
#pragma unroll
  for (int i = 0; i < vec_size; ++i) {
    const uint8_t packed = src_u8[i / 2];
    const uint8_t q = (i % 2 == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
    tgt[i] = __hfma(__ushort2half_rn(q), scale, zero);
  }
}

}  // namespace dequant
}  // namespace ftka

#endif  // FTKA_DEQUANTIZE_CUH_
