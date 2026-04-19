# triton_kernels

Triton kernels — the **third backend path** alongside CUDA (CuTe DSL)
and AscendC. Each kernel is a single `.py` file with a `@triton.jit`
entry and a thin host wrapper. No CMake, no `OBJECT` lib — Triton
JIT-compiles at first call.

## Why Triton lives here, not in `csrc/kernels/`

`csrc/` is for code that goes through `nvcc` or `ccec`. Triton kernels
go through the Triton compiler, which is a Python + LLVM toolchain
owned by the runtime, not CMake. Mixing them breaks the build model,
so Triton gets its own top-level directory.

## Why Triton at all

Two properties nothing else on this box offers:

1. **Same source runs on CUDA and Ascend.** Upstream Triton targets
   NVIDIA; `triton-ascend` (Huawei's fork / upstream backend) targets
   AscendC. One kernel.py, two hardware targets. For memory-bound ops
   we don't need to write the op twice in two different DSLs.
2. **Good fit for memory-bound kernels.** No tensor-core ceremony,
   SMEM tiling is one `tl.load(block_ptr)` call, quantize / pack /
   reduce in a few lines. The kernels where CuTe DSL is overkill (AI
   well below B200's ~281 FLOP/B ridge) belong here.

## Library choice matrix

| Op class                                  | Library      | Reason |
|-------------------------------------------|--------------|--------|
| Main `gemm_w4a4` — scaled NVFP4 / INT4    | **CuTe DSL** | compute-bound, needs `tcgen05` scaled-MMA, TMEM, 2-CTA; CUDA-only anyway |
| `quantize_w4a4_act_fuse_lora` etc.        | **Triton**   | memory-bound (AI ≪ ridge, B200 bench: 25–120 FLOP/B), plus Ascend needs coverage too |
| AscendC-only GEMM                         | **AscendC**  | no CuTe DSL on NPU; AscendC's cube unit is the native compute path |

Memory-bound decision rule: measure AI on B200 (see
`tmp/bench_lora_down.py` for a template). If AI is below ~1/3 of the
FP16 tensor-core ridge (~90 FLOP/B) **and** the op needs to run on
both CUDA and Ascend, Triton wins. Otherwise go native per backend.

## Per-op layout

```
triton_kernels/<op>/
    kernel.py     # @triton.jit kernel + torch-tensor host wrapper
    README.md     # op contract + where nunchaku does the same thing
```

## Toolchain

- Upstream Triton (CUDA): whatever torch's bundled version is, or the
  checkout at `tmp/triton` for development.
- `triton-ascend` (NPU): installed separately on the Ascend host; same
  Python API, different backend. Not required for local edit / CUDA
  builds.

The Ascend path ISN'T wired up yet — this directory exists today to
document the intent. First kernel to land is
`quantize_w4a4_act_fuse_lora`; its Ascend validation follows once we
have an NPU host with `triton-ascend`.
