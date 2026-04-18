#include "quantize_act_int4.h"

namespace svdquant::ascend {

void quantize_act_int4(const QuantizeActInt4Params& p, void* stream) {
    (void)p;
    (void)stream;
    // TODO: vector-unit reduction for per-row amax; pack to int4 on-chip.
}

}  // namespace svdquant::ascend
