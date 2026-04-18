# CUDA backend

## Target SMs

Declared in `cmake/cuda_arch.cmake`. Defaults:

| SM  | Arch     | Representative parts |
|-----|----------|----------------------|
| 80  | Ampere   | A100                 |
| 89  | Ada      | 4090, L40            |
| 90  | Hopper   | H100, H200           |
| 100 | Blackwell| B200 (reserved)      |

Override with `-DSVDQUANT_CUDA_ARCHS="80;90"`. Each listed arch also
gets a `SVDQUANT_HAS_SM<N>=1` compile define, so files can opt in per
arch without the build system knowing about them.

## Per-pod layout

```
csrc/kernels/<op>/cuda/
    kernel.cu           # top-level launcher; dispatches by capability
    sm80.cu             # (added when real kernels land)
    sm89.cu
    sm90.cu
```

The scaffold only ships `kernel.cu` with a host-side stub; per-SM
variants land as real implementations arrive.

## Build

```
CUDA=ON ASCEND=OFF ./scripts/build.sh
```

or directly:

```
cmake -S . -B build -G Ninja \
    -DSVDQUANT_ENABLE_CUDA=ON \
    -DSVDQUANT_ENABLE_ASCEND=OFF
cmake --build build
```

## Conventions

- Launch signatures take `void* stream` rather than `cudaStream_t` to
  keep the header free of CUDA includes — cast inside `kernel.cu`.
- `TensorRef::data` is a raw device pointer (`T*` cast from
  `cudaMalloc`/PyTorch storage).
- Kernels should prefer CUTLASS / cuBLAS primitives where they win;
  bespoke kernels are for shapes those libraries don't cover well.
