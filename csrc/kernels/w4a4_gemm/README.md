# w4a4_gemm

INT4 weight × INT4 activation matmul with per-token activation scale and
per-group weight scale. This is the quantized main path of the SVDQuant
linear layer. The residual low-rank term is a separate op
([`lowrank_branch`](../lowrank_branch)); fusion is a third op
([`fused_svdquant_linear`](../fused_svdquant_linear)).

## Public header
`include/w4a4_gemm.h` — `W4A4GemmParams`, plus `svdquant::cuda::w4a4_gemm`
and `svdquant::ascend::w4a4_gemm` launch declarations.

## Backend notes
- **cuda/** — per-SM variants land here (`sm80.cu` / `sm89.cu` / `sm90.cu`)
  once the first real kernel is written. Today it's a single stub.
- **ascend/** — host launcher only. AscendC device kernels (`__aicore__`
  code) will come in `ascend/kernel_device.cpp`, compiled by `ccec`.

## Baseline
Reference implementation (pure Python / PyTorch) will live under
`baseline/kernels/w4a4_gemm/` and is the numerical ground truth for tests.
