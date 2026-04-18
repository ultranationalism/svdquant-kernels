# Ascend NPU backend

Huawei CANN / AscendC. Other NPU vendors (Cambricon, etc.) are out of
scope for this repo; add them as sibling directories
(`csrc/kernels/<op>/<vendor>/`) if needed later.

## Toolchain

- **CANN toolkit** at `${ASCEND_HOME_PATH}` (default
  `/usr/local/Ascend/ascend-toolkit/latest`).
- `ccec` — AscendC device-side compiler (for `__aicore__` code).
- Host-side ACL runtime (`libascendcl`, `libruntime`).

Before configuring CMake:

```
source scripts/env_ascend.sh
```

This sources the CANN environment (`setenv.bash` / `set_env.sh`).
`cmake/FindCANN.cmake` then locates headers, libs, and `ccec`.

## Per-pod layout

```
csrc/kernels/<op>/ascend/
    kernel.cpp                # host launcher (plain C++)
    kernel_device.cpp         # (added later) __aicore__ kernel, compiled by ccec
```

`kernel.cpp` is compiled by the host C++ compiler and today is a stub.
When the first real AscendC kernel lands, we add a second file for the
device code and a custom build rule that feeds it to `ccec` and links
the resulting object into the pod's OBJECT library.

## Build

```
source scripts/env_ascend.sh
CUDA=OFF ASCEND=ON ./scripts/build.sh
```

or directly:

```
cmake -S . -B build -G Ninja \
    -DSVDQUANT_ENABLE_CUDA=OFF \
    -DSVDQUANT_ENABLE_ASCEND=ON
cmake --build build
```

## Conventions

- Launch signatures take `void* stream`; cast to `aclrtStream` inside
  `kernel.cpp`.
- `TensorRef::data` is a device address (what `aclrtMalloc` returns).
- Kernels should use AscendC's tiling helpers rather than hand-rolling
  DMA; cube unit for GEMM-shaped math, vector unit for elementwise /
  reductions.
