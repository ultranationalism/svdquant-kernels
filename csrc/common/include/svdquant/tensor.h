#pragma once
// Minimal backend-independent tensor descriptor.
//
// `data` is an opaque device pointer — its real type depends on the backend
// launcher (CUdeviceptr / aclrt device address / raw T*). Strides are in
// elements, not bytes. For sub-byte dtypes (int4) the caller is responsible
// for stride alignment.

#include <cstddef>
#include <cstdint>

#include "svdquant/dtype.h"

namespace svdquant {

constexpr int kMaxTensorDims = 4;

struct TensorRef {
    void*        data;
    DType        dtype;
    std::int32_t ndim;
    std::int64_t shape[kMaxTensorDims];
    std::int64_t stride[kMaxTensorDims];
};

}  // namespace svdquant
