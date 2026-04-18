# Per-kernel design notes

One markdown file per op as kernels mature, e.g.:

- `w4a4_gemm.md` — tile shape, scale handling, per-SM dispatch logic.
- `fused_svdquant_linear.md` — epilogue fusion, register budget.

Keep the op's `csrc/kernels/<op>/README.md` as a short orientation page;
put the actual design writeup (shape analysis, memory traffic, tuning
notes, wins and regressions) here.
