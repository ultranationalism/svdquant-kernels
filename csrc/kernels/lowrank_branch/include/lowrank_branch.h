#pragma once
// Low-rank correction branch — out = x @ L1 @ L2.
//
// L1 / L2 are the rank-r factors that reconstruct the quantization error
// absorbed out of the weight before INT4 quantization. Because r is small
// (typically 16-32), this is a pair of tall-skinny / skinny-tall matmuls
// and benefits from being implemented separately from the W4A4 main path
// unless fused (see `fused_svdquant_linear`).
//
// Logical shapes:
//   x    [M, K]
//   l1   [K, R]
//   l2   [R, N]
//   out  [M, N]

#include "svdquant/tensor.h"

namespace svdquant {

struct LowRankBranchParams {
    TensorRef x;
    TensorRef l1;
    TensorRef l2;
    TensorRef out;
};

namespace cuda   { void lowrank_branch(const LowRankBranchParams& p, void* stream); }
namespace ascend { void lowrank_branch(const LowRankBranchParams& p, void* stream); }

}  // namespace svdquant
