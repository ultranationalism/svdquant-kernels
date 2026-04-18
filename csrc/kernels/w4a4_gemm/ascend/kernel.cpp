#include "w4a4_gemm.h"

// Host-side launcher for the AscendC w4a4_gemm kernel.
//
// The actual device kernel (with __aicore__ qualifiers) will live alongside
// this file as `kernel_device.cpp` and be compiled by `ccec` — the build
// rule for that is wired in when the first real kernel lands. Until then
// this file is a host-side stub that keeps the symbol link-visible.

namespace svdquant::ascend {

void w4a4_gemm(const W4A4GemmParams& p, void* stream) {
    (void)p;
    (void)stream;
    // TODO: aclrtLaunchKernel(...) to dispatch the AscendC kernel.
}

}  // namespace svdquant::ascend
