// gemm_w4a4 — Ascend __aicore__ device kernel.
//
// Phase 1a (current): empty placeholder. CANN's ascendc.cmake compiles
// this source three times (precompile, AIC, AIV) and emits an
// auto-generated wrapper that calls `<our-symbol>_origin`. The
// wrapper is the same in all three passes, so this function must be
// *defined* in every pass — we can't gate it behind cube/vec
// preprocessor branches at this stage.
//
// In Phase 2 we'll switch to FA's pattern: keep one unconditional
// __global__ entry point but split work via `if constexpr (DAV_CUBE)`
// / `if constexpr (DAV_VEC)` inside the body, where DAV_CUBE/DAV_VEC
// are constexpr bools derived from `__DAV_CUBE__`/`__DAV_VEC__`.
// See `pto-isa/tests/npu/a2a3/src/st/testcase/tfa/tfa_kernel.cpp:51-61`.

#include "kernel_operator.h"

extern "C" __global__ [aicore] void
svdquant_gemm_w4a4_kernel(GM_ADDR params) {
    (void)params;
}
