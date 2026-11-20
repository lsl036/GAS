// Adaption for low-bit data types (8bit/4bit).
//
// Modified from:
//  https://github.com/efeslab/Atom/blob/main/kernels/include/flashinfer/quantization.cuh

#ifndef FTKA_INT4_CUH_
#define FTKA_INT4_CUH_

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <type_traits>

#include "flashinfer/vec_dtypes.cuh"

using namespace flashinfer;

namespace ftka {
namespace lowbit {

/*!
 * \brief Identifier used for 4bit quantization,
 *   for data size calculattion.
 */
struct uint_4bit {};

/*!
 * \brief Simliar to sizeof
 * \tparam T Data type to be sizeof
 */
template <typename T>
FLASHINFER_INLINE constexpr float size_of_type() {
  if constexpr (std::is_same_v<T, uint_4bit>) {
    return 0.5f;
  }
  return sizeof(T);
}

/*!
 * \brief Used to get the pointer by offset.
 * \tparam T A template indicates the data type
 * \param ptr Pointer to the data
 * \param offset Offset to the pointer
 */
template <typename T>
FLASHINFER_INLINE T* get_ptr(T* ptr, const size_t offset) {
  if constexpr (std::is_same_v<T, uint_4bit>) {
    return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(ptr) + offset / 2);
  } else if constexpr (std::is_same_v<T, const uint_4bit>) {
    // Patch for const qualifiers
    return reinterpret_cast<T*>(reinterpret_cast<const uint8_t*>(ptr) + offset / 2);
  } else {
    return ptr + offset;
  }
}

}  // namespace lowbit
}  // namespace ftka

/******************* vec_t<uint8_t> *******************/

namespace flashinfer {

#if !defined(FLASHINFER_HAS_UINT8_VEC_T)
// uint8_t x 1
template <>
struct vec_t<uint8_t, 1> {
  uint8_t data;

  FLASHINFER_INLINE uint8_t& operator[](size_t i) { return ((uint8_t*)(&data))[i]; }
  FLASHINFER_INLINE const uint8_t& operator[](size_t i) const {
    return ((const uint8_t*)(&data))[i];
  }
  FLASHINFER_INLINE uint8_t* ptr() { return reinterpret_cast<uint8_t*>(&data); }
  FLASHINFER_INLINE void fill(uint8_t val);
  FLASHINFER_INLINE void load(const uint8_t* ptr);
  FLASHINFER_INLINE void store(uint8_t* ptr) const;
  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, 1>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }

  FLASHINFER_INLINE static void memcpy(uint8_t* dst, const uint8_t* src);
};

// uint8_t x 4 or more
template <size_t vec_size>
struct vec_t<uint8_t, vec_size> {
  uint32_t data[vec_size / 4];

  FLASHINFER_INLINE uint8_t& operator[](size_t i) { return ((uint8_t*)(&data))[i]; }
  FLASHINFER_INLINE const uint8_t& operator[](size_t i) const {
    return ((const uint8_t*)(&data))[i];
  }
  FLASHINFER_INLINE uint8_t* ptr() { return reinterpret_cast<uint8_t*>(&data); }
  FLASHINFER_INLINE void fill(uint8_t val) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      data[i] = (val << 24) | (val << 16) | (val << 8) | val;
    }
  }

  FLASHINFER_INLINE void load(const uint8_t* ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      data[i] = ((uint32_t*)ptr)[i];
    }
  }

  FLASHINFER_INLINE void store(uint8_t* ptr) const {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      ((uint32_t*)ptr)[i] = data[i];
    }
  }

  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, vec_size>& src) {
    cast_from_impl(*this, src);
  }

  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }

  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }

  FLASHINFER_INLINE void memcpy(uint8_t* dst, const uint8_t* src) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      ((uint32_t*)dst)[i] = ((uint32_t*)src)[i];
    }
  }
};

// uint8_t x 4 or more
template <size_t vec_size>
struct vec_t<int16_t, vec_size> {
  uint32_t data[vec_size / 2];

  FLASHINFER_INLINE int16_t& operator[](size_t i) { return ((int16_t*)(&data))[i]; }
  FLASHINFER_INLINE const int16_t& operator[](size_t i) const {
    return ((const int16_t*)(&data))[i];
  }
  FLASHINFER_INLINE int16_t* ptr() { return reinterpret_cast<int16_t*>(&data); }
  FLASHINFER_INLINE void fill(int16_t val) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 2; ++i) {
      data[i] = (val << 16) | val;
    }
  }

  FLASHINFER_INLINE void load(const int16_t* ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 2; ++i) {
      data[i] = ((uint32_t*)ptr)[i];
    }
  }

  FLASHINFER_INLINE void store(int16_t* ptr) const {
#pragma unroll
    for (size_t i = 0; i < vec_size / 2; ++i) {
      ((uint32_t*)ptr)[i] = data[i];
    }
  }

  template <typename T>
  FLASHINFER_INLINE void cast_from(const vec_t<T, vec_size>& src) {
    cast_from_impl(*this, src);
  }

  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }

  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }

  FLASHINFER_INLINE void memcpy(int16_t* dst, const int16_t* src) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 2; ++i) {
      ((uint32_t*)dst)[i] = ((uint32_t*)src)[i];
    }
  }
};
#endif

/******************* vec_t<uint_4bit> *******************/

// uint_4bitx 8
// 32 bits. Used for decrease the computation load.
template <>
struct vec_t<ftka::lowbit::uint_4bit, 8> {
  uint32_t data[1];
  FLASHINFER_INLINE void load(const ftka::lowbit::uint_4bit* ptr) { data[0] = ((uint32_t*)ptr)[0]; }

  // Get the i-th element of packed uint32_t register
  FLASHINFER_INLINE uint32_t& at(size_t i) { return ((uint32_t*)data)[i]; }

  FLASHINFER_INLINE const uint32_t& at(size_t i) const { return ((uint32_t*)data)[i]; }

  FLASHINFER_INLINE ftka::lowbit::uint_4bit* ptr() {
    return reinterpret_cast<ftka::lowbit::uint_4bit*>(&data);
  }
};

// uint_4bitx 16
// 64 bits. Used for decrease the computation load.
template <>
struct vec_t<ftka::lowbit::uint_4bit, 16> {
  uint32_t data[2];
  FLASHINFER_INLINE void load(const ftka::lowbit::uint_4bit* ptr) {
    data[0] = ((uint32_t*)ptr)[0];
    data[1] = ((uint32_t*)ptr)[1];
  }

  // Get the i-th element of packed uint32_t register
  FLASHINFER_INLINE uint32_t& at(size_t i) { return ((uint32_t*)data)[i]; }

  FLASHINFER_INLINE const uint32_t& at(size_t i) const { return ((uint32_t*)data)[i]; }

  FLASHINFER_INLINE ftka::lowbit::uint_4bit* ptr() {
    return reinterpret_cast<ftka::lowbit::uint_4bit*>(&data);
  }
};

// uint_4bit x 32 or more

template <size_t vec_size>
struct vec_t<ftka::lowbit::uint_4bit, vec_size> {
  uint4 data[vec_size / 32];

  FLASHINFER_INLINE void load(const ftka::lowbit::uint_4bit* ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 32; ++i) {
      data[i] = ((uint4*)ptr)[i];
    }
  }

  // Get the i-th element of packed uint32_t register
  FLASHINFER_INLINE uint32_t& at(size_t i) { return ((uint32_t*)data)[i]; }

  FLASHINFER_INLINE const uint32_t& at(size_t i) const { return ((uint32_t*)data)[i]; }

  FLASHINFER_INLINE static void memcpy(ftka::lowbit::uint_4bit* dst,
                                       const ftka::lowbit::uint_4bit* src) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 32; ++i) {
      ((uint4*)dst)[i] = ((uint4*)src)[i];
    }
  }

  FLASHINFER_INLINE ftka::lowbit::uint_4bit* ptr() {
    return reinterpret_cast<ftka::lowbit::uint_4bit*>(&data);
  }
};

/******************* vec_t type cast *******************/

// NOTE(chaofan): This is just for pass the compilation, not used for real.

template <>
struct vec_cast<float, uint8_t> {
  template <size_t vec_size>
  FLASHINFER_INLINE static void cast(float* dst, const uint8_t* src) {
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      dst[i] = static_cast<float>(src[i]);
    }
  }
};

template <>
struct vec_cast<half, uint8_t> {
  template <size_t vec_size>
  FLASHINFER_INLINE static void cast(half* dst, const uint8_t* src) {
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      dst[i] = static_cast<half>(src[i]);
    }
  }
};

template <>
struct vec_cast<float, ftka::lowbit::uint_4bit> {
  template <size_t vec_size>
  FLASHINFER_INLINE static void cast(float* dst, const ftka::lowbit::uint_4bit* src) {
    const uint8_t* byte_ptr = reinterpret_cast<const uint8_t*>(src);
    for (size_t i = 0; i < vec_size; i += 2) {
      dst[i] = static_cast<float>((byte_ptr[i / 2] >> 4) & 0x0F);
      dst[i + 1] = static_cast<float>(byte_ptr[i / 2] & 0x0F);
    }
  }
};

template <>
struct vec_cast<half, ftka::lowbit::uint_4bit> {
  template <size_t vec_size>
  FLASHINFER_INLINE static void cast(half* dst, const ftka::lowbit::uint_4bit* src) {
    const uint8_t* byte_ptr = reinterpret_cast<const uint8_t*>(src);
    for (size_t i = 0; i < vec_size; i += 2) {
      dst[i] = static_cast<float>((byte_ptr[i / 2] >> 4) & 0x0F);
      dst[i + 1] = static_cast<float>(byte_ptr[i / 2] & 0x0F);
    }
  }
};

}  // namespace flashinfer

#endif  // FTKA_INT4_CUH_
