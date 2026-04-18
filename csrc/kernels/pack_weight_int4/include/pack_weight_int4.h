#pragma once
// Offline weight packer — FP weight → INT4 packed + per-group scale.
//
// Group quantization along K with group size G:
//   scale[n, g] = amax(w[n, gG : gG + G]) / 7
//   q[n, k]     = clamp(round(w[n, k] / scale[n, k / G]), -8, 7)
//
// The output `out_q` layout is backend-specific — CUDA reorders into a
// layout friendly to its MMA tile; AscendC reorders for cube-unit loads.
// Offline step, called once per weight at model load.
//
// Logical shapes:
//   w        [N, K]        fp16 / bf16 / fp32
//   out_q    [N, K]        int4 packed, backend-specific interleave
//   scale    [N, K / G]    fp16 / fp32

#include "svdquant/tensor.h"

namespace svdquant {

struct PackWeightInt4Params {
    TensorRef w;
    TensorRef out_q;
    TensorRef scale;
    int       group_size;
};

namespace cuda   { void pack_weight_int4(const PackWeightInt4Params& p, void* stream); }
namespace ascend { void pack_weight_int4(const PackWeightInt4Params& p, void* stream); }

}  // namespace svdquant
