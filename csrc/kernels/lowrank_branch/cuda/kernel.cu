#include "lowrank_branch.h"

namespace svdquant::cuda {

void lowrank_branch(const LowRankBranchParams& p, void* stream) {
    (void)p;
    (void)stream;
    // TODO: two back-to-back GEMMs (x @ L1, then @ L2) — likely reuse
    // cuBLAS for the outer call and a bespoke kernel only if the shape
    // punishes cuBLAS (small R, skinny N).
}

}  // namespace svdquant::cuda
