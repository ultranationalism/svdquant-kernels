#pragma once
// W4A4 GEMM — INT4 weight × INT4 activation matmul with per-token activation
// scale and per-group weight scale. This is the quantized main path of the
// SVDQuant linear layer; the low-rank correction lives in a separate pod.
//
// Logical shapes (packed layout is backend-specific):
//   a        [M, K]        int4  per-token act, two nibbles / byte
//   w        [N, K]        int4  per-group weight
//   a_scale  [M]           fp16 / fp32
//   w_scale  [N, K / G]    fp16 / fp32   (G == group_size)
//   out      [M, N]        fp16 / bf16

#include "svdquant/tensor.h"

namespace svdquant {

struct W4A4GemmParams {
    TensorRef a;
    TensorRef w;
    TensorRef a_scale;
    TensorRef w_scale;
    TensorRef out;
    int       group_size;
};

namespace cuda   { void w4a4_gemm(const W4A4GemmParams& p, void* stream); }
namespace ascend { void w4a4_gemm(const W4A4GemmParams& p, void* stream); }

}  // namespace svdquant
