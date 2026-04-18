# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Execution environment

This machine is a **cross-compile / edit host only**. There are no GPUs
or NPUs attached; `nvcc`, `ccec`, `nvidia-smi`, `npu-smi`, and friends
are **not** available and will not be available. All kernel runs and
profiling/tracing happen on remote **serverless GPU/NPU instances**.

Consequences for how you work here:

- **Do not try to build to verify.** A local `cmake`/`./scripts/build.sh`
  will fail or disable both backends. Parse-level verification stops at
  reading the generated CMake; anything further is done on the remote
  instance.
- **Do not try to run or bench kernels locally.** No devices. If a
  workflow needs "run + measure", that step belongs on the serverless
  side — script it, don't attempt it here.
- **Scratch artifacts go in `tmp/`** (gitignored): trace dumps pulled
  back from remote, generated sources, one-off scripts, staging
  tarballs. Don't pollute the repo root.

## What this repo is (and isn't)

A **kernel development workbench** for SVDQuant (W4A4 linear with low-rank
correction), targeting both NVIDIA GPUs (CUDA) and Huawei Ascend NPUs
(AscendC) from one source tree.

It is explicitly **not** a shipping library. Consequences that affect how
you work here:

- **No runtime dispatcher.** Callers pick `svdquant::cuda::<op>` or
  `svdquant::ascend::<op>` directly. Do not add a router. If you think
  one is needed, you're in the wrong repo — that belongs to whoever
  integrates these kernels into a framework.
- **No Python bindings (yet).** PyTorch appears only under `baseline/` as
  numerical ground truth. Do not pull `torch` into `csrc/` or
  `svdquant_kernels/`. Bindings come later, after kernels stabilize.
- **Pods are OBJECT libs, not a single `.so`.** Linking into a shared
  library is an integration concern.

Current state: scaffold. Every pod's `kernel.cu` / `kernel.cpp` is a
host-side stub so the build wires end-to-end before any real kernel lands.

## Build

Both backends are probed independently and each is an `option()`. Missing
toolchains downgrade to a warning, not an error — so on a CUDA-only box
Ascend silently disables, and vice versa.

```
# defaults: both backends on, tests/bench off
./scripts/build.sh

# CUDA only / Ascend only
CUDA=ON ASCEND=OFF ./scripts/build.sh
CUDA=OFF ASCEND=ON ./scripts/build.sh

# turn on tests/bench (empty today — opt-in adds the subdirs)
TESTS=ON BENCH=ON ./scripts/build.sh

# Debug build
BUILD_TYPE=Debug ./scripts/build.sh
```

Before configuring on an Ascend machine: `source scripts/env_ascend.sh`
(sources CANN's `setenv.bash` / `set_env.sh`). Override the CANN
location with `ASCEND_HOME_PATH=...`.

Direct CMake flags if you need them:
`-DSVDQUANT_ENABLE_CUDA`, `-DSVDQUANT_ENABLE_ASCEND`,
`-DSVDQUANT_BUILD_TESTS`, `-DSVDQUANT_BUILD_BENCHMARKS`,
`-DSVDQUANT_CUDA_ARCHS="80;90"`.

No lint/test commands exist yet — the repo is scaffold-only.

## Adding a new kernel pod

One line in `csrc/kernels/CMakeLists.txt`:

```cmake
svdquant_add_kernel_pod(<op>)
```

Pod layout the helper expects:

```
csrc/kernels/<op>/
  include/<op>.h        # struct <Op>Params; svdquant::{cuda,ascend}::<op> decls
  cuda/kernel.cu        # optional — skipped if missing
  ascend/kernel.cpp     # optional — skipped if missing
  README.md
```

The helper globs by exact filename (`cuda/kernel.cu`, `ascend/kernel.cpp`).
Extra per-SM files (e.g. `cuda/sm90.cu`) should be added from inside
`kernel.cu` via `#include` or wired into the helper as the need arises.

## Conventions that aren't obvious from the code

- **Op signatures live in the pod's `include/<op>.h`** — one header per
  op, declaring a `<Op>Params` struct and two launch functions with
  identical signatures: `svdquant::cuda::<op>(params, void* stream)` and
  `svdquant::ascend::<op>(params, void* stream)`. Keep these two in
  sync; divergence defeats the point of the workbench.
- **`void* stream`, not `cudaStream_t` / `aclrtStream`.** Keeps the
  public header free of backend includes. Cast inside `kernel.cu` /
  `kernel.cpp`.
- **`TensorRef::data` is opaque.** Each backend interprets it as its own
  device pointer type. Strides are in elements, not bytes.
- **AscendC split:** `ascend/kernel.cpp` is the host launcher, compiled
  by the host C++ compiler. Real device code (`__aicore__`) goes into a
  sibling file (convention: `ascend/kernel_device.cpp`) compiled by
  `ccec`. That `ccec` rule is **not wired yet** — add it the first time
  a pod needs real device code, not before.
- **Per-SM CUDA variants:** new SM numbers go into `SVDQUANT_CUDA_ARCHS`
  in `cmake/cuda_arch.cmake`. Each listed arch auto-gets a
  `SVDQUANT_HAS_SM<N>=1` compile define so source files can opt in
  without touching CMake.
- **`baseline/` is PyTorch-only and must stay importable without any
  kernel having built.** It's ground truth for tests — readability
  beats speed, no `torch.compile`, no fused ops, no backend tricks.

## The five ops (why they exist as separate pods)

SVDQuant factors `y = x @ W` as
`dequant(int4(x) @ int4(W')) + x @ L1 @ L2` where `W = W' + L1 @ L2`.

- `pack_weight_int4` — offline: `W'` → INT4 + group scale.
- `quantize_act_int4` — online: per-token INT4 activation.
- `w4a4_gemm` — INT4 × INT4 main path.
- `lowrank_branch` — `x @ L1 @ L2` residual.
- `fused_svdquant_linear` — the three online ops fused; the shipping path.

The unfused trio exists so each piece can be developed and numerically
verified in isolation against its baseline. `fused_svdquant_linear`
should reuse those kernels' logic (not their launches), and will
outperform calling them back-to-back by hiding the low-rank branch under
the quantized GEMM's epilogue.
