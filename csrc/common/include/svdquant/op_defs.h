#pragma once
// SVDQuant operator catalog (index).
//
// Each operator pod under csrc/kernels/<op>/ owns its own public header with
// the concrete host-callable launch signature. This file is the directory:
//
//   w4a4_gemm              csrc/kernels/w4a4_gemm/include/w4a4_gemm.h
//   lowrank_branch         csrc/kernels/lowrank_branch/include/lowrank_branch.h
//   fused_svdquant_linear  csrc/kernels/fused_svdquant_linear/include/fused_svdquant_linear.h
//   quantize_act_int4      csrc/kernels/quantize_act_int4/include/quantize_act_int4.h
//   pack_weight_int4       csrc/kernels/pack_weight_int4/include/pack_weight_int4.h
//
// Every pod declares two backend namespaces:
//
//   namespace svdquant::cuda   { ... }
//   namespace svdquant::ascend { ... }
//
// with identical launch signatures. Dispatch to a specific backend is the
// caller's responsibility — there is no runtime router in this repo, which
// is intentional: this is a kernel development workbench, not a product.
