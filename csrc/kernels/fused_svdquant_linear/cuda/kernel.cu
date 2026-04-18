#include "fused_svdquant_linear.h"

namespace svdquant::cuda {

void fused_svdquant_linear(const FusedSvdquantLinearParams& p, void* stream) {
    (void)p;
    (void)stream;
    // TODO: fused path — quantize activation in-register, run W4A4 mainloop,
    // stream low-rank branch into the same epilogue.
}

}  // namespace svdquant::cuda
