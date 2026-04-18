#include "pack_weight_int4.h"

namespace svdquant::ascend {

void pack_weight_int4(const PackWeightInt4Params& p, void* stream) {
    (void)p;
    (void)stream;
    // TODO: group-amax scale, quantize, interleave into the cube-unit
    // load layout expected by ascend w4a4_gemm.
}

}  // namespace svdquant::ascend
