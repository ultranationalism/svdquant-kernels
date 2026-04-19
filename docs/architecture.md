# Architecture

`svdquant-kernels` is a **kernel development workbench**, not a shipping
library. The shape of the repo follows from that:

- Each operator is a self-contained **pod**. Native pods (compiled
  by nvcc or ccec) live under `csrc/kernels/<op>/`; Triton pods
  (JIT-compiled by the Triton runtime, shared across CUDA and Ascend)
  live under `triton_kernels/<op>/`.
- Pods are independent — building one doesn't require any other to
  compile. Adding a native pod is one line in
  `csrc/kernels/CMakeLists.txt`; adding a Triton pod is just dropping
  a directory under `triton_kernels/`.
- **No runtime dispatch.** The caller picks `svdquant::cuda::<op>`,
  `svdquant::ascend::<op>`, or a Triton `@triton.jit` entry directly
  at the call site. Dispatch belongs to whoever integrates these
  kernels into a framework — that's explicitly out of scope here.
- **No Python bindings on the native pods (yet).** The native pods
  expose C++ launchers only; PyTorch ground truth lives in
  `baseline/`. Triton pods are themselves Python, so their call site
  is torch-tensor native. Real `torch.library` bindings around the
  C++ ops come later, once kernels stabilize.

## Backends

| Backend    | Directory                         | Compiler                        | Toolchain doc           |
|------------|-----------------------------------|---------------------------------|-------------------------|
| CUDA       | `csrc/kernels/<op>/cuda/`         | `nvcc` (CuTe DSL)               | [gpu.md](./gpu.md)      |
| Ascend     | `csrc/kernels/<op>/ascend/`       | C++ host + `ccec`               | [npu.md](./npu.md)      |
| Triton     | `triton_kernels/<op>/kernel.py`   | upstream Triton (CUDA) + `triton-ascend` (NPU) | [gpu.md](./gpu.md), [npu.md](./npu.md) |

Native pods declare `svdquant::cuda::<op>` and `svdquant::ascend::<op>`
with identical signatures, expressed in terms of backend-agnostic
`TensorRef` from `csrc/common/include/svdquant/tensor.h`. The meaning
of `TensorRef::data` is backend-specific.

Triton pods expose a single Python function (typed against
`torch.Tensor`) that both backends call with the same signature.

## Library choice per op

Decision rule: **compute-bound + CUDA-only → CuTe DSL**; **compute-bound
+ NPU-only → AscendC**; **memory-bound + cross-backend → Triton**.
"Memory-bound" here means measured arithmetic intensity well below
B200's FP16 tensor-core ridge (~281 FLOP/B) — concretely, below ~90
FLOP/B. See `tmp/bench_lora_down.py` for the template benchmark.

| Op                            | Library      | Measured AI (ZImage Turbo) | Why |
|-------------------------------|--------------|----------------------------|-----|
| `gemm_w4a4`                   | CuTe DSL + AscendC | several hundred FLOP/B     | `tcgen05` scaled-MMA + TMEM + 2-CTA tiles are the whole point |
| `quantize_w4a4_act_fuse_lora` | Triton       | 26–120 FLOP/B              | memory-bound (AI ≪ ridge); needs Ascend coverage; cuBLAS leaves ~60% of HBM on the table at small K/R |

## The two ops (why these)

The online W4A4 linear chain from nunchaku's public API
(`tmp/nunchaku/src/kernels/zgemm/gemm_w4a4.cu:34-105,113-125`) is
exactly two kernels:

- `quantize_w4a4_act_fuse_lora` (Triton, preprocess) —
  quantize the next layer's input to NVFP4 + produce its LoRA-down
  projection (`x @ L1`).
- `gemm_w4a4` (native, compute) — scaled-MMA main path + LoRA-up
  residual + bias + optional next-layer quantize.

Weight packing (`W'` → INT4/NVFP4 + block scales) is offline and
one-shot, so it lives as a pure-Python utility under `baseline/`
rather than as a kernel pod. TMA re-tiles a contiguous packed layout
cheaply at load time on SM_100/SM_103, so a GEMM-tile-specific disk
format buys nothing.

Activation quantization has no standalone caller in the nunchaku
dataflow — it's always fused into `quantize_w4a4_act_fuse_lora`
(pre-GEMM) or into a previous `gemm_w4a4`'s epilogue (post-GEMM),
never both.

## Build model

- Top-level `CMakeLists.txt` probes CUDA and CANN; each is an
  independent `option()` and either can be disabled. The repo builds
  fine with only one backend enabled.
- Each native pod compiles to an `OBJECT` library `svdquant::<op>`.
  They are **not** linked into a single `.so` by default — that's an
  integration concern, not a workbench concern.
- Triton pods don't go through CMake at all. They JIT on first call.
- Tests and benchmarks are opt-in via
  `-DSVDQUANT_BUILD_TESTS=ON` / `-DSVDQUANT_BUILD_BENCHMARKS=ON`.
