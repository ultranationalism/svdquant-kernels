# svdquant-kernels

SVDQuant (W4A4 + low-rank correction) kernels **for vLLM**. The goal
is to back vLLM's diffusion path with 4-bit kernels on both NVIDIA
Blackwell (SM_100/103, **NVFP4**) and Huawei Ascend NPUs (**INT4**)
from a single source tree. Each backend uses the 4-bit format its
tensor unit actually supports.

**This is a kernel development workbench**, not yet a library. There
is no Python dispatcher and no framework integration yet — each kernel
is built, tested, and profiled in isolation. vLLM bindings land after
the kernels stabilize; fusions that would require vLLM pipeline
changes (e.g., folding a preceding GLU into a quantize op) are
explicitly out of scope so the kernels stay drop-in.

## Layout

```
csrc/
  common/                  headers shared across backends (dtype, TensorRef, macros)
  kernels/                 native pods — nvcc (CuTe DSL) or ccec (AscendC)
    <op>/
      include/<op>.h       op's public header (backend-agnostic signature)
      cuda/kernel.cu       CUDA implementation
      ascend/kernel.cpp    AscendC host launcher (device code added later)
      README.md
triton_kernels/            Triton pods — one source, runs on CUDA + Ascend
  <op>/
    kernel.py              @triton.jit + torch-tensor host wrapper
    README.md
baseline/                  PyTorch reference implementations (numerical ground truth)
tests/                     per-op numerical correctness tests
benchmarks/                per-op micro-benchmarks
docs/                      architecture.md, gpu.md, npu.md, per-kernel notes
cmake/                     FindCANN.cmake, cuda_arch.cmake
scripts/                   build.sh, env_ascend.sh
```

## The two ops

A W4A4→W4A4 linear chain is two kernels, mirroring nunchaku's public
C++ API:

| Pod                                                      | Location               | Library   | Role |
|----------------------------------------------------------|------------------------|-----------|------|
| [`gemm_w4a4`](csrc/kernels/gemm_w4a4/)                   | `csrc/kernels/`        | CuTe DSL (CUDA) / AscendC (NPU) | Main W4A4 scaled-MMA + LoRA-up residual + bias + optional next-layer quantize |
| [`quantize_w4a4_act_fuse_lora`](triton_kernels/quantize_w4a4_act_fuse_lora/) | `triton_kernels/`      | Triton    | Memory-bound preprocessing: NVFP4 quantize of input + `x @ lora_down` small GEMM |

Library choice: compute-bound ops that need `tcgen05` / TMEM / 2-CTA
go native per backend; memory-bound ops that need to run on both CUDA
and Ascend go through Triton (one `.py` source, two hardware targets).

Weight packing (FP → INT4/NVFP4 + block scales) is offline and lives
as a pure-Python utility in `baseline/` — no kernel pod.

## Build

```
# CUDA only
CUDA=ON ASCEND=OFF ./scripts/build.sh

# Ascend only
source scripts/env_ascend.sh
CUDA=OFF ASCEND=ON ./scripts/build.sh

# Both (each autodetected; missing toolchain disables its backend with a warning)
./scripts/build.sh
```

See [`docs/architecture.md`](docs/architecture.md),
[`docs/gpu.md`](docs/gpu.md), and [`docs/npu.md`](docs/npu.md) for
per-backend details.

## Status

Scaffold. No kernels yet — every pod currently holds a host-side stub so
the build is wired end-to-end before the first real kernel lands.

## License

Apache-2.0
