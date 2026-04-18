#include "w4a4_gemm.h"

namespace svdquant::cuda {

// TODO: dispatch on compute capability and land per-arch kernels:
//   SM80  — Ampere ldmatrix + IMMA.s32.s4.s4 (stub)
//   SM89  — Ada    DP4A / INT4 IMMA variant  (stub)
//   SM90  — Hopper WGMMA + TMA               (stub)
void w4a4_gemm(const W4A4GemmParams& p, void* stream) {
    (void)p;
    (void)stream;
}

}  // namespace svdquant::cuda
