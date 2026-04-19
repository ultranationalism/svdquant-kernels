#include "gemm_w4a4.h"

// AscendC implementation lands here. Cube unit for the main W4A4
// MMA, vector unit for the epilogue (bias, lora_up accumulate,
// optional next-layer quantize). Device-side `__aicore__` code goes
// in a sibling `kernel_device.cpp` (compiled by `ccec`) once needed.

namespace svdquant::ascend {

void gemm_w4a4(const GemmW4A4Params& p, void* stream) {
    (void)p;
    (void)stream;
}

}  // namespace svdquant::ascend
