#include "gemm_w4a4.h"

// CuTe DSL implementation lands here. For SM_100 / SM_103 the
// compute path is `tcgen05.mma.kind::mxf4nvf4.block_scale.scale_vec::4X`
// with FP32 accumulator and FP8 UE4M3 block scales; epilogue fuses
// bias + wcscales + `lora_act_in @ lora_up` + optional next-layer
// NVFP4 quantize.
//
// Until the real kernel lands, this is a host-side stub so the
// symbol links and the build pipeline stays honest.

namespace svdquant::cuda {

void gemm_w4a4(const GemmW4A4Params& p, void* stream) {
    (void)p;
    (void)stream;
}

}  // namespace svdquant::cuda
