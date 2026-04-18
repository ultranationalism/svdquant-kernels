#include "fused_svdquant_linear.h"

namespace svdquant::ascend {

void fused_svdquant_linear(const FusedSvdquantLinearParams& p, void* stream) {
    (void)p;
    (void)stream;
    // TODO: fused AscendC kernel — pack act-quant, w4a4 mainloop, and the
    // low-rank add into a single pipeline across cube + vector units.
}

}  // namespace svdquant::ascend
