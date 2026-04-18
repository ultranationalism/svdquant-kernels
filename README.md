# svdquant-kernels

Cross-architecture kernels for SVDQuant (W4A4 with low-rank correction).
Targets both NVIDIA GPUs (CUDA) and Huawei Ascend NPUs (AscendC) from a
single source tree.

**This is a kernel development workbench**, not a production library.
There is no Python dispatcher and no framework integration — each kernel
is built, tested, and profiled in isolation. Framework bindings
(PyTorch etc.) come later.

## Layout

```
csrc/
  common/                  headers shared across backends (dtype, TensorRef, macros)
  kernels/
    <op>/
      include/<op>.h       op's public header (backend-agnostic signature)
      cuda/kernel.cu       CUDA implementation
      ascend/kernel.cpp    AscendC host launcher (device code added later)
      README.md
baseline/                  PyTorch reference implementations (numerical ground truth)
tests/                     per-op numerical correctness tests
benchmarks/                per-op micro-benchmarks
docs/                      architecture.md, gpu.md, npu.md, per-kernel notes
cmake/                     FindCANN.cmake, cuda_arch.cmake
scripts/                   build.sh, env_ascend.sh
```

## The five ops

SVDQuant decomposes `y = x @ W` as
`dequant(int4(x) @ int4(W')) + x @ L1 @ L2`. The pods mirror that:

| Pod                       | Role                                           |
|---------------------------|------------------------------------------------|
| `pack_weight_int4`        | Offline: FP weight → INT4 + per-group scale    |
| `quantize_act_int4`       | Online: dynamic per-token INT4 activation      |
| `w4a4_gemm`               | INT4 × INT4 GEMM with scales (main quant path) |
| `lowrank_branch`          | `x @ L1 @ L2` residual                         |
| `fused_svdquant_linear`   | The three online ops fused end-to-end          |

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
