#include "pack_weight_int4.h"

namespace svdquant::cuda {

void pack_weight_int4(const PackWeightInt4Params& p, void* stream) {
    (void)p;
    (void)stream;
    // TODO: group-amax scale, quantize, interleave into the MMA-friendly
    // layout expected by w4a4_gemm on the same SM.
}

}  // namespace svdquant::cuda
