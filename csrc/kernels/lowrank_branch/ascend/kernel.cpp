#include "lowrank_branch.h"

namespace svdquant::ascend {

void lowrank_branch(const LowRankBranchParams& p, void* stream) {
    (void)p;
    (void)stream;
    // TODO: launch AscendC kernel(s). Two sequential matmuls; first pass
    // may fit the AI core's cube unit, second pass may prefer vector unit
    // depending on R.
}

}  // namespace svdquant::ascend
