# gemm_w4a4

Main W4A4 linear — compute-bound, tensor-core-first. Consumes
pre-quantized NVFP4 activation + weight (produced by
`triton_kernels/quantize_w4a4_act_fuse_lora/` on the previous layer),
runs scaled-MMA, accumulates the SVDQuant low-rank residual, and
(optionally) produces pre-quantized output for the next layer.

Signature: see `include/gemm_w4a4.h`.

CUDA path: CuTe DSL over `tcgen05.mma.kind::mxf4nvf4` on SM_100 /
SM_103. Ascend path: AscendC cube unit (not wired yet).

Reference: `tmp/nunchaku/src/kernels/zgemm/gemm_w4a4.cu:34-105`
(nunchaku's host launcher — we omit the attention-fusion parameters
`norm_q/norm_k/rotary_emb/out_q/out_k/out_v/out_vk/out_linearattn/poolout`
since those belong to the attention kernel, not the linear).
