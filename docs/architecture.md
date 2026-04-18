# Architecture

`svdquant-kernels` is a **kernel development workbench**, not a shipping
library. The shape of the repo follows from that:

- Each operator is a self-contained **pod** under `csrc/kernels/<op>/`
  with its own public header and one source file per backend.
- Pods are independent — building one doesn't require any other to
  compile, and adding a pod is one line in `csrc/kernels/CMakeLists.txt`.
- **No runtime dispatch.** The caller picks `svdquant::cuda::<op>` or
  `svdquant::ascend::<op>` at the call site. Dispatch belongs to whoever
  integrates these kernels into a framework — that's explicitly out of
  scope here.
- **No Python bindings (yet).** PyTorch is only pulled in under
  `baseline/` as numerical ground truth. Real `torch.library` bindings
  come later, once kernels stabilize.

## Backends

| Backend | Directory                  | Compiler              | Toolchain doc          |
|---------|----------------------------|-----------------------|------------------------|
| CUDA    | `csrc/kernels/<op>/cuda/`  | `nvcc`                | [gpu.md](./gpu.md)     |
| Ascend  | `csrc/kernels/<op>/ascend/`| C++ host + `ccec`     | [npu.md](./npu.md)     |

Each pod declares `svdquant::cuda::<op>` and `svdquant::ascend::<op>` with
identical signatures, expressed in terms of backend-agnostic
`TensorRef` from `csrc/common/include/svdquant/tensor.h`. The meaning of
`TensorRef::data` is backend-specific.

## The five ops (why these)

SVDQuant decomposes a linear `y = x @ W` as
`y ≈ dequant(int4(x) @ int4(W')) + x @ L1 @ L2` where `W = W' + L1 @ L2`.
The pods line up with the pieces of that decomposition:

- `pack_weight_int4` — offline: produce `int4(W')` and its scales.
- `quantize_act_int4` — online: produce `int4(x)` and its per-token scale.
- `w4a4_gemm` — online: `int4(x) @ int4(W')` main quantized path.
- `lowrank_branch` — online: `x @ L1 @ L2` residual.
- `fused_svdquant_linear` — production path: the three online ops fused.

The unfused pods exist to develop and correctness-check each piece in
isolation; the fused pod is what you'd actually call.

## Build model

- Top-level `CMakeLists.txt` probes CUDA and CANN; each is an
  independent `option()` and either can be disabled. The repo builds
  fine with only one backend enabled.
- Each pod compiles to an `OBJECT` library `svdquant::<op>`. They are
  **not** linked into a single `.so` by default — that's an integration
  concern, not a workbench concern.
- Tests and benchmarks are opt-in via
  `-DSVDQUANT_BUILD_TESTS=ON` / `-DSVDQUANT_BUILD_BENCHMARKS=ON`.
