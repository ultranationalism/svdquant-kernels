#pragma once
// SVDQuant operator catalog (index).
//
// Each native operator pod under csrc/kernels/<op>/ owns its own
// public header with the concrete host-callable launch signature.
// Triton-based ops aren't declared here — they're Python entries
// under triton_kernels/<op>/.
//
//   Native (CuTe DSL / AscendC):
//     gemm_w4a4                 csrc/kernels/gemm_w4a4/include/gemm_w4a4.h
//
//   Triton (CUDA + Ascend via triton-ascend, same source):
//     quantize_w4a4_act_fuse_lora  triton_kernels/quantize_w4a4_act_fuse_lora/kernel.py
//
// Weight packing (FP → INT4/NVFP4 + block scales) is an offline,
// one-shot calibration step; it lives as a pure-Python utility in
// baseline/, not as a kernel pod — TMA makes on-device tile re-layout
// cheap enough that baking a tile layout into the packed format buys
// nothing.
//
// Every native pod declares two backend namespaces:
//
//   namespace svdquant::cuda   { ... }
//   namespace svdquant::ascend { ... }
//
// with identical launch signatures. Dispatch to a specific backend is
// the caller's responsibility — there is no runtime router in this
// repo, which is intentional: this is a kernel development workbench,
// not a product.
