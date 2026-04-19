# CUDA backend

## Target SMs

Declared in `cmake/cuda_arch.cmake`. Scope is intentionally narrow:

| SM   | Arch      | Representative parts     |
|------|-----------|--------------------------|
| 100  | Blackwell | B100 / B200              |
| 103  | Blackwell | data-center Blackwell variants |

Everything else (Turing through Hopper, plus consumer Blackwell
SM_120a/121a) is covered by `nunchaku`; this repo exists to fill the
SM_100/SM_103 gap, not duplicate that work. See
`tmp/nunchaku/setup.py:41-64` for nunchaku's arch list.

Override with `-DSVDQUANT_CUDA_ARCHS="100;103"`. Each listed arch
also gets a `SVDQUANT_HAS_SM<N>=1` compile define, so files can opt
in per arch without the build system knowing about them.

## Per-pod layout

```
csrc/kernels/<op>/cuda/
    kernel.cu           # top-level launcher; dispatches by capability
    sm100.cu            # (added when real kernels land)
    sm103.cu
```

The scaffold only ships `kernel.cu` with a host-side stub; per-SM
variants land as real implementations arrive. Real kernels on this
path use **CuTe DSL** (CUTLASS 3.x) for `tcgen05.mma` scaled-MMA
variants — that's what B200's tensor cores speak.

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
- Kernels in `csrc/kernels/` should use CUTLASS 3.x / CuTe DSL
  primitives; bespoke hand-rolled CUDA is for shapes CuTe can't
  cover well.

## When to pick Triton instead

If an op is memory-bound on B200 (AI well below the ~281 FLOP/B FP16
tensor-core ridge) AND needs to also run on Ascend NPU, put it under
`triton_kernels/<op>/` instead — one `kernel.py` runs on both
backends (upstream Triton for CUDA, `triton-ascend` for NPU). See
`../triton_kernels/README.md` for the library-choice rule.
