# pack_weight_int4

Offline weight packer. Quantizes an FP weight matrix to group-wise INT4
along K, producing an `out_q` tensor in the backend-specific interleaved
layout that its `w4a4_gemm` expects.

```
scale[n, g] = amax(w[n, gG : gG + G]) / 7
q[n, k]     = clamp(round(w[n, k] / scale[n, k / G]), -8, 7)
```

Called once per weight at model load — correctness matters far more than
speed here; the packed layout must exactly match the GEMM's expected
load pattern.

## Public header
`include/pack_weight_int4.h`

## Backend notes
- **cuda/** — interleave for MMA tile (SM-specific; see `w4a4_gemm`).
- **ascend/** — interleave for cube-unit loads.
