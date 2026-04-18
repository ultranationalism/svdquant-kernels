#pragma once
// Full SVDQuant linear, fused:
//
//   (a_q, a_scale) = quantize_act_int4(x)
//   y_main         = w4a4_gemm(a_q, w, a_scale, w_scale)
//   y_lr           = x @ L1 @ L2
//   out            = y_main + y_lr
//
// All done as a single kernel (or a tight pipeline of kernels) to hide the
// low-rank branch under the quantized GEMM. This is the shipping path; the
// unfused pods exist for standalone development and baselining.
//
// Logical shapes:
//   x         [M, K]       fp16 / bf16
//   w         [N, K]       int4 packed, per-group scale along K
//   w_scale   [N, K / G]
//   l1        [K, R]       fp16 / bf16
//   l2        [R, N]       fp16 / bf16
//   out       [M, N]       fp16 / bf16

#include "svdquant/tensor.h"

namespace svdquant {

struct FusedSvdquantLinearParams {
    TensorRef x;
    TensorRef w;
    TensorRef w_scale;
    TensorRef l1;
    TensorRef l2;
    TensorRef out;
    int       group_size;
};

namespace cuda   { void fused_svdquant_linear(const FusedSvdquantLinearParams& p, void* stream); }
namespace ascend { void fused_svdquant_linear(const FusedSvdquantLinearParams& p, void* stream); }

}  // namespace svdquant
