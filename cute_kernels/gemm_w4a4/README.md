# gemm_w4a4 (CuTe DSL, CUDA)

Main SVDQuant W4A4 linear on Blackwell SM_100 / SM_103 — NVFP4 scaled
MMA (`tcgen05.mma.kind::mxf4nvf4.block_scale.scale_vec::4X`) + LoRA
low-rank residual (β interleaved into the main K-loop) + optional
per-channel affine and next-layer NVFP4 quantize.

**Design**: `docs/kernels/gemm_w4a4.md` (ten sections covering β
justification, interleave pattern, tmem / smem budget, warp roles,
tiler choice, epilogue, skeleton source strategy, staged rollout).

**Contract**: `launch(...)` in `kernel.py`. Torch tensors at the host
boundary; NVFP4 is the uint8-packed layout produced by the preceding
`triton_kernels/quantize_w4a4_act_fuse_lora/` op.

**Reference math**: `baseline/kernels/gemm_w4a4/ref.py` (pure PyTorch
fp32 ground truth — what this kernel must match per `tmp/smoke_gemm.py`).

**Staging**:

| version | scope                                                        |
| ------- | ------------------------------------------------------------ |
| v0      | main NVFP4 only, no LoRA, no wcscales, no bias               |
| v1      | + LoRA β-interleaved per design §2                           |
| v2      | + per-col `* wcscales + bias` epilogue                       |
| v3      | + optional next-layer NVFP4 quantize                         |

Currently at v1 (task #34).

**Reference skeleton**: `tmp/cutlass_bs_gemm_persistent.py`
(vendored copy of CUTLASS `examples/python/CuTeDSL/blackwell/
dense_blockscaled_gemm_persistent.py`) — stable-API scaled-MMA +
persistent scheduler. The 1021-line
`experimental/blackwell/dense_block_scaled_gemm.py` is cleaner but
its `cute.experimental` API is hard-gated to CUDA 13.1+; we're on
CUDA 13.0 (both local and Modal's B200 image), so we base off the
stable persistent example.

## Baseline — CUTLASS NVFP4 on same B200 / same shapes

This is the honest ceiling. CUTLASS's own `dense_blockscaled_gemm_
persistent.py` (main NVFP4 MMA, no LoRA / no epilogue scale / no
next-quant — strictly the same op our v0 does) vs our kernel on
`GEMM_SHAPES` in fp16 out. Run with `modal run
scripts/modal_app.py::cutlass_nvfp4_bench`. MFU vs 10 PFLOPS B200
dense NVFP4 peak.

| shape (M, K, N)       | CUTLASS 1-CTA 128×256 | CUTLASS 2-CTA 256×128 | CUTLASS 2-CTA 256×256 | ours 1-CTA        | ours 2-CTA Phase 1 |
| --------------------- | --------------------- | --------------------- | --------------------- | ----------------- | ------------------ |
|  256 × 3840 × 3072    |   564 TF  5.6%        |   734 TF  7.3%        |   588 TF  5.9%        |    98 TF  1.0%    |   100 TF  1.0%     |
| 4352 × 3840 × 3072    |  3847 TF 38.5%        |  4202 TF 42.0%        |  4545 TF 45.4%        |  1309 TF 13.1%    |  1185 TF 11.8%     |
| 4352 × 3840 × 15360   |  4167 TF 41.7%        |  5181 TF 51.8%        |  5836 TF 58.4%        |  2735 TF 27.4%    |  2599 TF 26.0%     |
| 4352 × 15360 × 3840   |  4096 TF 41.0%        |  5903 TF 59.0%        |  6339 TF 63.4%        |  2646 TF 26.5%    |  2964 TF 29.6%     |
| 4352 × 10240 × 3072   |  4174 TF 41.7%        |  5375 TF 53.8%        |  6074 TF 60.7%        |  2299 TF 23.0%    |  2350 TF 23.5%     |

Takeaways:

- **Real NVFP4 ceiling on this HW ≈ 60% MFU** (CUTLASS 2-CTA
  256×256). Not the 30-40% that I had been quoting from memory.
- **1-CTA gap**: CUTLASS ≈ 41% MFU, ours ≈ 27%. At the same tile
  (128×256). Missing pieces are on our side — persistent scheduler,
  stage count, epilogue / MMA overlap. This is task #41.
- **2-CTA Phase 1 gap**: CUTLASS 2-CTA 256×128 hits ≈ 53-59% even
  though FLOPs/atom equals 1-CTA 128×256. Ours gets essentially
  zero 2-CTA benefit (≈ 28% vs 27%). So Phase 1's ~0 speedup is not
  inherent — it's our implementation. Phase 2 (256×256 +
  overlapping_accum, task #39) should target ~60% to match CUTLASS.
- Small-M (M=256) is grid-limited for both — 12 tiles vs 148 SMs,
  tensor cores idle most of the kernel. Don't over-read that row.
