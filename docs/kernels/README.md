# Per-kernel design notes

One markdown file per op as kernels mature, e.g.:

- `gemm_w4a4.md` — `tcgen05` scaled-MMA tile shape, scale handling,
  LoRA-up epilogue register budget, optional next-layer quantize
  fusion.
- `quantize_w4a4_act_fuse_lora.md` — Triton SMEM tiling, LoRA-down
  reduction plan, NVFP4 E2M1 packing layout, per-16-K-block FP8
  scale write path.

Keep the op's pod-level README (`csrc/kernels/<op>/README.md` or
`triton_kernels/<op>/README.md`) as a short orientation page with the
contract and the reference anchor; put the actual design writeup
(shape analysis, memory traffic, tuning notes, wins and regressions)
here.
