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
  static constexpr int VEC_WIDTH = 8;
  if (!std::is_same_v<src_type, lowbit::uint_4bit>) {
    // NOTE(chaofan): to pass the compilation
    return;
  }
  static_assert(vec_size % VEC_WIDTH == 0, "Invalid vector size");

  uint32_t const* src_ptr = reinterpret_cast<uint32_t const*>(src);

#pragma unroll
  for (int i = 0; i < vec_size; i += VEC_WIDTH) {
    uint32_t h[4];                                    // 128bits = 8 * fp16
    uint32_t const i4s = *(src_ptr + i / VEC_WIDTH);  // 8 * int4_t = uint32_t

    // First, we extract the i4s and construct an intermediate fp16 number.
    static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint32_t BOTTOM_MASK = 0x000f000f;
    static constexpr uint32_t TOP_MASK = 0x00f000f0;
    static constexpr uint32_t I4s_TO_F16s_MAGIC_NUM = 0x64006400;

    // Note that the entire sequence only requires 1 shift instruction. This is thanks to the
    // register packing format and the fact that we force our integers to be unsigned, and account
    // for this in the fp16 subtractions. In addition, I exploit the fact that sub and fma have the
    // same throughput in order to convert elt_23 and elt_67 to fp16 without having to shift them to
    // the bottom bits before hand.

    // Shift right by 8 to now consider elt_45 and elt_67. Issue first to hide RAW dependency if we
    // issue immediately before required.
    const uint32_t top_i4s = i4s >> 8;
    // Extract elt_01 - (i4s & 0x000f000f) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[0])
                 : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    // Extract elt_23 (i4s & 0x00f000f0) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[1])
                 : "r"(i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    // Extract elt_45 (top_i4s & 0x000f000f) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[2])
                 : "r"(top_i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    // Extract elt_67 (top_i4s & 0x00f000f0) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[3])
                 : "r"(top_i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));

    // I use inline PTX below because I am not sure if the compiler will emit float2half
    // instructions if I use the half2 ctor. In this case, I chose performance reliability over code
    // readability.

    // This is the half2 {1032, 1032} represented as an integer.
    static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64086408;
    // This is the half2 {1 / 16, 1 / 16} represented as an integer.
    static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00;
    // This is the half2 {-72, -72} represented as an integer.
    static constexpr uint32_t NEG_72 = 0xd480d480;

    // Finally, we construct the output numbers.
    // Convert elt_01
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
    // Convert elt_23
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                 : "=r"(h[1])
                 : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_72));
    // Convert elt_45
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[2]) : "r"(h[2]), "r"(FP16_TOP_MAGIC_NUM));
    // Convert elt_67
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                 : "=r"(h[3])
                 : "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(NEG_72));

    // half2* h_half2 = reinterpret_cast<half2*>(h);
    // float2* tgt_ptr = reinterpret_cast<float2*>(tgt.ptr() + i);
    // tgt_ptr[0] = __half22float2(h_half2[0]);
    // tgt_ptr[1] = __half22float2(h_half2[1]);
    // tgt_ptr[2] = __half22float2(h_half2[2]);
    // tgt_ptr[3] = __half22float2(h_half2[3]);

    // Scale & Zero
    half2 scale2 = make_half2(scale, scale);
    half2 zero2 = make_half2(zero, zero);
    half2* result = reinterpret_cast<half2*>(h);

    // asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[0]) : "r"(scale2), "r"(zero2),
    // "r"(h[0])); asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[1]) : "r"(scale2),
    // "r"(zero2), "r"(h[1])); asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[2]) :
    // "r"(scale2), "r"(zero2), "r"(h[2])); asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" :
    // "=r"(h[3]) : "r"(scale2), "r"(zero2), "r"(h[3]));

    result[0] = __hfma2(result[0], scale2, zero2);
    result[1] = __hfma2(result[1], scale2, zero2);
    result[2] = __hfma2(result[2], scale2, zero2);
    result[3] = __hfma2(result[3], scale2, zero2);
  }
}

}  // namespace dequant
}  // namespace ftka

#endif  // FTKA_DEQUANTIZE_CUH_
