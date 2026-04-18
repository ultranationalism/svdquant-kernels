#pragma once
// Backend-independent dtype tag. Each backend maps these to its native
// storage type (e.g. __half / __nv_bfloat16 on CUDA, half / bfloat16 on
// AscendC). Ground truth for layout/stride interpretation lives in tensor.h.

#include <cstdint>

namespace svdquant {

enum class DType : std::uint8_t {
    kFloat16  = 0,
    kBFloat16 = 1,
    kFloat32  = 2,
    kInt4     = 3,  // packed, two nibbles per byte, little-endian within byte
    kInt8     = 4,
    kInt32    = 5,
};

constexpr int element_bits(DType dt) {
    switch (dt) {
        case DType::kFloat16:  return 16;
        case DType::kBFloat16: return 16;
        case DType::kFloat32:  return 32;
        case DType::kInt4:     return 4;
        case DType::kInt8:     return 8;
        case DType::kInt32:    return 32;
    }
    return 0;
}

}  // namespace svdquant
