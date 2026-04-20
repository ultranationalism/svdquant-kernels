#pragma once
// gemm_w4a4 — main SVDQuant W4A4 linear kernel (Ascend C-side header).
//
// This header declares the **Ascend-side** C++ entry point only. The
// CUDA path does not go through C++ at all — it is a CuTe DSL kernel
// under `cute_kernels/gemm_w4a4/kernel.py`, called directly from
// Python with torch tensors. Keep this header free of CUDA decls.
//
// Consumes packed INT4 activation + FP16 block scales, packed INT4
// weight + FP16 block scales, runs AscendC cube GEMM, accumulates the
// SVDQuant low-rank residual `lora_act_in @ lora_up` in the epilogue,
// optionally biases / rescales, and optionally re-quantizes the result
// for the next layer. (The CUDA / NVFP4 math is the same at the
// tensor-shape level; only the 4-bit format and the tensor-unit
// language differ — see CLAUDE.md "4-bit format splits by backend".)
//
// This is the compute-bound half of SVDQuant. The memory-bound
// preprocessing op that produces `act` / `ascales` / `lora_act_in`
// lives in `triton_kernels/quantize_w4a4_act_fuse_lora/` (Triton,
// shared across CUDA and Ascend).
//
// Logical shapes (Ascend INT4 packed layout; strides in elements):
//   act          [M,   K/2]   uint8     2 signed INT4 / byte
//   wgt          [N,   K/2]   uint8     2 signed INT4 / byte
//   ascales      [K/64, M]    fp16      per-64-K-block act scale
//   wscales      [K/64, N]    fp16      per-64-K-block weight scale
//   lora_act_in  [M,   R]     fp32      = fpsum @ lora_down from previous op
//   lora_up      [N,   R]     fp16/bf16
//   bias         [N]          fp16/bf16 (optional)
//   smooth       [N_next]     fp16/bf16 (optional; next layer's smooth factor)
//   out          [M, N]       fp16/bf16
//   qout         [M, N/2]     uint8     (optional) pre-quantized output for next layer
//   oscales      [N/64, M]    fp16      (optional) matching scales for `qout`
//
// Reference: `tmp/nunchaku/src/kernels/zgemm/gemm_w4a4.cu:34-105`
// (the `gemm_w4a4` host launcher; we intentionally omit nunchaku's
// attention-fusion parameters — norm/rotary/QKV-split belong to the
// attention kernel, not the linear).

#include "svdquant/tensor.h"

namespace svdquant {

struct GemmW4A4Params {
    // Inputs
    TensorRef act;
    TensorRef wgt;
    TensorRef ascales;
    TensorRef wscales;

    // Low-rank residual
    TensorRef lora_act_in;
    TensorRef lora_up;

    // Optional per-channel affine on the main output
    TensorRef bias;       // may be .data == nullptr
    float     alpha;      // per-tensor weight scale multiplier

    // Main output
    TensorRef out;

    // Optional fused next-layer quantize (produces inputs for the
    // next gemm_w4a4). All three must be valid together or all null.
    TensorRef qout;
    TensorRef oscales;
    TensorRef smooth;
};

namespace ascend { void gemm_w4a4(const GemmW4A4Params& p, void* stream); }

}  // namespace svdquant
