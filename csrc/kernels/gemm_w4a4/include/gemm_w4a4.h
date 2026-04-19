#pragma once
// gemm_w4a4 — main SVDQuant W4A4 linear kernel.
//
// Consumes packed NVFP4 (or INT4) activation + FP8 block scales,
// packed NVFP4 (or INT4) weight + FP8 block scales, runs
// `tcgen05`-based scaled-MMA (CUDA SM_100/SM_103) or AscendC cube
// GEMM (NPU), accumulates the SVDQuant low-rank residual
// `lora_act_in @ lora_up` in the epilogue, optionally biases /
// rescales, and optionally re-quantizes the result for the next
// layer.
//
// This is the compute-bound half of SVDQuant. The memory-bound
// preprocessing op that produces `act` / `ascales` / `lora_act_in`
// lives in `triton_kernels/quantize_w4a4_act_fuse_lora/` (Triton,
// shared across CUDA and Ascend).
//
// Logical shapes (packed layout is backend-specific; strides in
// elements):
//   act          [M,   K/2]   uint8     2 NVFP4 E2M1 / byte
//   wgt          [N,   K/2]   uint8     2 NVFP4 E2M1 / byte
//   ascales      [K/16, M]    fp8_e4m3  per-16-K-block, one scale/block
//   wscales      [K/16, N]    fp8_e4m3  per-16-K-block, one scale/block
//   lora_act_in  [M,   R]     fp32      = fpsum @ lora_down from previous op
//   lora_up      [N,   R]     fp16/bf16
//   bias         [N]          fp16/bf16 (optional)
//   wcscales     [N]          fp16/bf16 (optional, NVFP4 path)
//   smooth       [N_next]     fp16/bf16 (optional; next layer's smooth factor)
//   out          [M, N]       fp16/bf16
//   qout         [M, N/2]     uint8     (optional) pre-quantized output for next layer
//   oscales      [N/16, M]    fp8_e4m3  (optional) matching scales for `qout`
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
    TensorRef wcscales;   // may be .data == nullptr (NVFP4 only)
    float     alpha;      // per-tensor weight scale multiplier

    // Main output
    TensorRef out;

    // Optional fused next-layer quantize (produces inputs for the
    // next gemm_w4a4). All three must be valid together or all null.
    TensorRef qout;
    TensorRef oscales;
    TensorRef smooth;

    bool use_fp4;   // true = NVFP4 tcgen05 path, false = INT4 path
};

namespace cuda   { void gemm_w4a4(const GemmW4A4Params& p, void* stream); }
namespace ascend { void gemm_w4a4(const GemmW4A4Params& p, void* stream); }

}  // namespace svdquant
