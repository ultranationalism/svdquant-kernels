# lowrank_branch

`out = x @ L1 @ L2` — the low-rank residual that reconstructs the
quantization error absorbed out of the weight before INT4 quantization.

Rank `r` is small (typically 16–32), so this is a pair of tall-skinny /
skinny-tall matmuls. Kept separate from `w4a4_gemm` to allow independent
tuning; the `fused_svdquant_linear` pod does the epilogue fusion.

## Public header
`include/lowrank_branch.h`

## Backend notes
- **cuda/** — likely a thin cuBLAS wrapper initially; bespoke kernel only
  if small-R / skinny shapes punish cuBLAS.
- **ascend/** — two sequential matmuls; cube unit for the first, vector
  unit possibly preferable for the second depending on R.
