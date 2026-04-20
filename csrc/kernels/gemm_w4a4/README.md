# gemm_w4a4 — Ascend pod

Main W4A4 linear — compute-bound, tensor-core-first. Consumes
pre-quantized INT4 activation + weight (produced by
`triton_kernels/quantize_w4a4_act_fuse_lora/` on the previous layer),
runs cube-unit GEMM, accumulates the SVDQuant low-rank residual, and
(optionally) produces pre-quantized output for the next layer.

This directory is the **Ascend** side of `gemm_w4a4`. The CUDA / NVFP4
side does not go through C++ — it is a CuTe DSL pod at
`cute_kernels/gemm_w4a4/kernel.py` (called directly from Python with
torch tensors). Formats split by backend: NVFP4 on CUDA, INT4 on
Ascend (see CLAUDE.md "4-bit format splits by backend"). Kept in the
same op name because the math, shapes, and dataflow are identical.

Signature: see `include/gemm_w4a4.h` (Ascend entry point only).

Reference: `tmp/nunchaku/src/kernels/zgemm/gemm_w4a4.cu:34-105`
(nunchaku's host launcher — we omit the attention-fusion parameters
`norm_q/norm_k/rotary_emb/out_q/out_k/out_v/out_vk/out_linearattn/poolout`
since those belong to the attention kernel, not the linear).
