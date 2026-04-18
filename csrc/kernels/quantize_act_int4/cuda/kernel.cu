#include "quantize_act_int4.h"

namespace svdquant::cuda {

void quantize_act_int4(const QuantizeActInt4Params& p, void* stream) {
    (void)p;
    (void)stream;
    // TODO: warp-reduce amax along K, compute scale, quantize-and-pack
    // two nibbles / byte in a single pass.
}

}  // namespace svdquant::cuda
