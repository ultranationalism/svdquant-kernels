# baseline

Numerical ground truth for every kernel pod — pure-Python / pure-PyTorch
implementations that mirror the C++ op signatures. **Not** a production
path; its only job is to generate reference outputs tests can diff
against.

Weight packing (FP → INT4/NVFP4 + block scales) also lives here as a
utility, even though no kernel pod mirrors it: packing is offline and
one-shot, so the Python function IS the implementation, not a ground
truth for something else. Tests consume it to produce packed inputs
for `gemm_w4a4` / `quantize_w4a4_act_fuse_lora`.

## Planned layout
```
baseline/
  kernels/
    gemm_w4a4/ref.py                   # fake-quant W4A4 linear (scaled MMA + LoRA-up + bias)
    quantize_w4a4_act_fuse_lora/ref.py # pre-GEMM quantize + LoRA-down
  pack/
    int4.py           # FP → INT4 + group scale (port of deepcompressor RTN)
    nvfp4.py          # FP → E2M1 + FP8 block scale (per 16-K block)
  __init__.py
```

## Rules
1. **Readability over speed** — the reference exists to be obviously correct.
2. **Match the C++ op signature exactly** — same tensor shapes, same
   dtypes, same group-size semantics.
3. **No backend-specific tricks** — no custom CUDA, no torch.compile, no
   fused ops. Straight `torch.matmul` / loops are fine.
4. Keep it standalone: importing `baseline.kernels.gemm_w4a4.ref` must
   not require any kernel build to have succeeded.

PyTorch is a dev-time dependency only; it will be added to
`pyproject.toml` when the first baseline lands.
