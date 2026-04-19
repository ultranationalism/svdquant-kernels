# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Execution environment

This machine is primarily a **cross-compile / edit host**. The full
CUDA 13 toolkit (`/usr/local/cuda-13.0`, `nvcc` 13.0.88) and CANN 8.5
toolkit (`ccec`, aarch64 `hcc` cross-toolchain) are installed, so
`cmake` + `./scripts/build.sh` succeeds locally. There is also a
**local consumer Blackwell GPU** (RTX 5060 Ti, SM_120) — enough to
JIT-run Triton kernels for correctness checks, **not** enough to
validate our native pods (they compile only for SM_100/103) or to
produce meaningful performance numbers. There is no local Ascend NPU.

Remote execution lives on two serverless channels:

- **CUDA / Blackwell B200 → Modal.** Build locally for `SM_100`, ship
  `build/` (or a tarball) into a Modal function tagged `gpu="B200"`.
  Entry point: `scripts/modal_app.py` via `modal run`.
- **Ascend NPU / Atlas A3 → OpenI (启智社区).** Cross-build aarch64
  artifacts locally, tar them, `scripts/ship.sh` uploads to the OpenI
  dataset; the serverless pod auto-extracts at boot.

### Local Python environment

**`/root/vllm-omni/.venv` is the canonical Python env for this repo.**
Python 3.12, torch 2.11 + cu130 (matches the host CUDA 13 toolkit),
triton 3.6. Use it for all Triton correctness runs, `baseline/`
reference code, and anything else under `tmp/` that needs to
import torch or triton:

```
/root/vllm-omni/.venv/bin/python tmp/smoke_lora_down.py
```

Do not use `/root/miniconda3` (base ships torch-cpu) or
`/root/miniconda3/envs/vllm` (torch is cu128, triton 3.5 — lags cu130
and will eventually drift). The repo itself has no Python packaging
yet; nothing is installed *into* the venv for this project — the venv
is just a known-good toolset we borrow.

### Consequences for how you work here

- **Local native build is a compile-check only.** Use it to catch type
  / symbol errors. `ctest` / bench / any runtime run of a native pod
  only makes sense remotely (SM_100/103).
- **Triton pods can be sanity-checked locally.** The SM_120 card is
  fine as a Triton correctness target; just remember perf tuning must
  happen on B200.
- `nvcc` comes from `/usr/local/cuda-13.0/bin` — the conda `base`
  environment ships a shadowing `nvcc` 12.8 first on PATH. `scripts/build.sh`
  scrubs the environment with `env -i` before configuring CMake; if you
  invoke CMake some other way, put `/usr/local/cuda` first in PATH or
  set `CUDAToolkit_ROOT` / `CMAKE_CUDA_COMPILER` explicitly.
- **Scratch artifacts go in `tmp/`** (gitignored): trace dumps pulled
  back from remote, generated sources, one-off scripts, staging
  tarballs. Don't pollute the repo root.

## What this repo is (and isn't)

A **kernel development workbench** for SVDQuant (W4A4 linear with low-rank
correction), **targeting vLLM's diffusion path**. Cross-architecture:
NVIDIA GPUs (CUDA, SM_100/SM_103 data-center Blackwell only — nunchaku
covers everything else, see `tmp/nunchaku/setup.py:41-64`) and Huawei
Ascend NPUs (AscendC) from one source tree.

Because vLLM is the consumer, kernels must be **drop-in at the linear
boundary**: `input → svdquant_op → output`. Fusions that would require
modifying vLLM's own pipeline (e.g., absorbing a preceding SwiGLU / GeGLU
gate into a quantize op, as nunchaku does with `fuse_glu`) are
**explicitly out of scope** — the perf win isn't worth the upstream
patch burden and the maintenance coupling. Keep the API narrow.

**4-bit format splits by backend**:
- **CUDA (SM_100/103)**: NVFP4 only (E2M1 values + 16-element FP8-E4M3
  block scales). Blackwell tcgen05 dropped INT4 scaled-MMA; CuTe DSL's
  `make_blockscaled_trivial_tiled_mma` only exposes MXF4/NVFP4/MXF8.
- **Ascend NPU**: INT4 (signed INT4 values + per-64-K-block FP16
  scales). Ascend's cube unit has INT4 MMA but no FP4. Math mirrors
  nunchaku's pre-Blackwell INT4 path.

So the native pods (`gemm_w4a4/cuda/*.cu` vs `gemm_w4a4/ascend/*.cpp`)
are naturally format-specialized — each backend uses what its tensor
unit actually supports. Triton pods (one source, two backends) branch
on `fp4: bool` at the Python host layer, which becomes a `tl.constexpr`
inside the kernel.

Kernels come in two flavors by compiler path:

- **Native** (`csrc/kernels/<op>/`) — nvcc (CuTe DSL) or ccec
  (AscendC). Used for compute-bound ops that need `tcgen05`
  scaled-MMA / TMEM / 2-CTA on CUDA, or the cube unit on Ascend.
- **Triton** (`triton_kernels/<op>/`) — one `kernel.py` JIT-compiled
  by upstream Triton on CUDA and by `triton-ascend` on NPU. Used for
  memory-bound ops that need to ship on both backends without writing
  the op twice.

Decision rule: if AI is well below B200's FP16 tensor-core ridge
(~281 FLOP/B) **and** the op needs Ascend coverage, it's a Triton
kernel. Otherwise it's native per backend.

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
`-DSVDQUANT_CUDA_ARCHS="100;103"` (default; data-center Blackwell only — see
`cmake/cuda_arch.cmake`).

No lint/test commands exist yet — the repo is scaffold-only.

## Adding a new kernel pod

### Native pod (CuTe DSL / AscendC)

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
Extra per-SM files (e.g. `cuda/sm100.cu`) should be added from inside
`kernel.cu` via `#include` or wired into the helper as the need arises.

### Triton pod (CUDA + Ascend)

No CMake line. Just drop a directory under `triton_kernels/`:

```
triton_kernels/<op>/
  kernel.py             # @triton.jit kernel + torch-tensor host wrapper
  README.md             # op contract + ref anchor (usually a nunchaku source line)
```

Triton JITs at first call; there's nothing to build ahead of time.
`triton-ascend` has to be installed on the NPU host for the Ascend
path — not required for local edit / CUDA runs.

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
- **Triton kernels in `triton_kernels/` are NOT tests or baselines.**
  They are real kernels that ship; the only difference from native
  pods is the compiler path (Triton → PTX/AscendC instead of nvcc).
  Treat them as production surface.

## The two ops (why this split)

A W4A4→W4A4 linear chain is two kernels, mirroring nunchaku's
public C++ API (`tmp/nunchaku/src/kernels/zgemm/gemm_w4a4.cu`):

```
[fp x]
  ↓ quantize_w4a4_act_fuse_lora(x, lora_down, smooth)               ← Triton
  ↓   → act_nvfp4 = quant(x / smooth)                                      [M, K/2]
  ↓   → lora_act_out = x @ lora_down                                       [M, R] fp32
  ↓   → ascales                                                            [K/16, M] fp8
[act, lora_act_in, ascales]
  ↓ gemm_w4a4(act, lora_act_in, wgt, lora_up, ascales, wscales,     ← CuTe DSL (CUDA)
  ↓           bias, wcscales, [qout, oscales, smooth_next])           ← AscendC (NPU)
  ↓   → y_fp = scaled_mma(act, wgt) · wcscale + bias
  ↓   → y_fp += lora_act_in @ lora_up
  ↓   → [optional] qout = quant(y_fp / smooth_next)                        for next layer
[y (, optional qout/oscales)]
```

Note: nunchaku's kernel has an extra `fuse_glu` flag that folds the
preceding SwiGLU (`gate ⊙ silu(value)`) into the load step. We do
**not** replicate it — see the "What this repo is" section above;
vLLM pipeline intrusion is out of scope.

- `gemm_w4a4` (native pod) — compute-bound scaled-MMA + LoRA-up
  residual + bias + optional next-layer quantize. The numbers that
  need `tcgen05` / `TMEM` / 2-CTA live here.
- `quantize_w4a4_act_fuse_lora` (Triton pod) — memory-bound
  preprocessing: quantize input + do the small `x @ lora_down`
  down-projection. Measured AI 26–120 FLOP/B on ZImage Turbo
  shapes — well below the ~281 FLOP/B B200 tensor-core ridge. Not
  worth the CuTe DSL ceremony, and Ascend needs it too.

**Weight packing is not a pod.** Converting FP weights to INT4/NVFP4 +
block scales is offline and one-shot — run once per model at calibration
time. It lives as a pure-Python utility under `baseline/`. TMA on
SM_100/SM_103 re-tiles a contiguous packed layout cheaply enough at
load time that baking a GEMM-tile-specific layout into the packed file
buys nothing. Don't add `pack_weight_*` back as a CUDA/AscendC pod.

**Activation quantization is not its own pod either.** It only ever
runs fused with the LoRA-down projection (the Triton op above) or as
the epilogue of a previous `gemm_w4a4` (its optional `qout`/`oscales`
outputs). A standalone `quantize_act_*` op has no caller in nunchaku's
dataflow.
