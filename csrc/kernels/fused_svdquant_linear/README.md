# fused_svdquant_linear

The shipping path. Fuses activation quantization, W4A4 GEMM, and the
low-rank correction branch into a single kernel (or a tight pipeline) so
the low-rank add hides under the quantized GEMM's math.

The unfused pods (`quantize_act_int4`, `w4a4_gemm`, `lowrank_branch`) are
kept around for standalone development, numerical baselining, and as the
building blocks this fused kernel will reuse.

## Public header
`include/fused_svdquant_linear.h`

## Backend notes
- **cuda/** — act-quant in-register; W4A4 mainloop; low-rank streamed
  through the epilogue.
- **ascend/** — fused pipeline across cube + vector units.
