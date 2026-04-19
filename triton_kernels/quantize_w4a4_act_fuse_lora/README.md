# quantize_w4a4_act_fuse_lora

Pre-GEMM preprocessing op for SVDQuant W4A4 linear: takes the FP
activation of the NEXT linear's input, produces three outputs in one
kernel:

1. Packed NVFP4 (or INT4) activation + per-16-K-block FP8 scales —
   consumed by the next `gemm_w4a4`.
2. LoRA-down activation: `input @ lora_down` → `[M, R]` fp32. The
   next linear's `gemm_w4a4` accumulates `lora_act @ lora_up` in its
   epilogue (the SVDQuant low-rank residual).
3. Optional SwiGLU / GLU gate fuse on the input before the quant pass.

## Math

```
fpsum[m, k] = input[m, k]                       # optionally: silu(x[:,:N]) * x[:,N:] for GLU
lora_act_out[m, r] = Σ_k fpsum[m, k] * lora_down[k, r]   # small GEMM, K→R
# output quantization, per-16-K-block:
for k_blk in blocks_of_16:
    s = max(|fpsum[m, k_blk] / smooth[k_blk]|) / 6.0
    oscales[m, k_blk // 16] = fp8_e4m3(s)
    qout[m, k_blk:k_blk+16]  = pack_nvfp4(fpsum[m, k_blk:k_blk+16] / smooth / s)
```

Smooth factor only divides the **quant** path; the LoRA-down path
consumes raw `fpsum`. That asymmetry is load-bearing — SVDQuant
math only permits smoothing on the branch where the weight scale is
pre-adjusted offline.

## Why Triton (not CuTe DSL)

Memory-bound on B200. Measured via `tmp/bench_lora_down.py`:

| shape (ZImage Turbo @ 1024², M=4352)  | AI (FLOP/B) | %HBM cuBLAS hits |
|---------------------------------------|-------------|------------------|
| K=3840,  R=32                         | ~26         | 30%              |
| K=3840,  R=128                        | ~96         | 42%              |
| K=10240, R=32                         | ~26         | 81%              |
| K=10240, R=128                        | ~99         | 74%              |

B200 FP16 tensor-core ridge ≈ 281 FLOP/B — none of these shapes
reach it. `tcgen05` / TMEM / 2-CTA buy nothing. Triton fuses the
matmul's `input` read with the quantize pass's `input` read, which a
pure matmul (cuBLAS) doesn't, so a tight Triton version can actually
beat the cuBLAS number above while also writing the packed output.

Second reason: this op needs an Ascend NPU path too. Triton
(`triton-ascend`) gives us one source for both.

## Reference implementation

- **C++ launcher** (nunchaku, SM_75–121a):
  `tmp/nunchaku/src/kernels/zgemm/gemm_w4a4.cuh:1096-1184`
  (`quantize_w4a4_fuse_lora_kernel`)
- **Python caller** (nunchaku):
  `tmp/nunchaku/nunchaku/models/linear.py:191-218`
  (`SVDQW4A4Linear.quantize`)
- **Reference math** (deepcompressor RTN fake-quant):
  `tmp/deepcompressor/deepcompressor/quantizer/kernel/rtn.py:68-117`

## Input / output contract (nunchaku-aligned)

```
input          : [M, actualN]      fp16/bf16  (actualN = 2·N if fuse_glu)
lora_down      : [N,      R]       fp16/bf16
smooth         : [N]               fp16/bf16  (optional)
----
output (qout)  : [M, N/2]          uint8      (2 nibbles/byte NVFP4 E2M1)
oscales        : [N/16,     M]     fp8_e4m3   (NVFP4) or fp16 (INT4, block=64)
lora_act_out   : [M,      R]       fp32
```

M is padded to `pad_size=256`. `N` is the in-features dim of the
next linear. R (LoRA rank) is a checkpoint-time constant; ZImage
Turbo ships R ∈ {32, 128, 256}.
