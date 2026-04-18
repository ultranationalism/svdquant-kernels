#pragma once
// Dynamic per-token INT4 activation quantization.
//
//   scale[m] = amax(x[m, :]) / 7           (per-token symmetric)
//   q[m, k]  = round(x[m, k] / scale[m])   clamped to [-8, 7]
//
// Two nibbles per byte; nibble 0 = element 2k, nibble 1 = element 2k+1.
// Layout detail may change per backend — see individual impls.
//
// Logical shapes:
//   x       [M, K]        fp16 / bf16
//   out_q   [M, K]        int4 packed
//   scale   [M]           fp16 / fp32

#include "svdquant/tensor.h"

namespace svdquant {

struct QuantizeActInt4Params {
    TensorRef x;
    TensorRef out_q;
    TensorRef scale;
};

namespace cuda   { void quantize_act_int4(const QuantizeActInt4Params& p, void* stream); }
namespace ascend { void quantize_act_int4(const QuantizeActInt4Params& p, void* stream); }

}  // namespace svdquant
