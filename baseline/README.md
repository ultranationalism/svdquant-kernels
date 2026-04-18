# baseline

Numerical ground truth for every kernel pod — pure-Python / pure-PyTorch
implementations that mirror the C++ op signatures. **Not** a production
path; its only job is to generate reference outputs tests can diff
against.

## Planned layout
```
baseline/
  kernels/
    w4a4_gemm/ref.py
    lowrank_branch/ref.py
    fused_svdquant_linear/ref.py
    quantize_act_int4/ref.py
    pack_weight_int4/ref.py
  __init__.py
```

## Rules
1. **Readability over speed** — the reference exists to be obviously correct.
2. **Match the C++ op signature exactly** — same tensor shapes, same
   dtypes, same group-size semantics.
3. **No backend-specific tricks** — no custom CUDA, no torch.compile, no
   fused ops. Straight `torch.matmul` / loops are fine.
4. Keep it standalone: importing `baseline.kernels.w4a4_gemm.ref` must
   not require any kernel build to have succeeded.

PyTorch is a dev-time dependency only; it will be added to
`pyproject.toml` when the first baseline lands.
