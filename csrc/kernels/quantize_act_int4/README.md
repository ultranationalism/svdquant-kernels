# quantize_act_int4

Dynamic per-token INT4 activation quantization.

```
scale[m] = amax(x[m, :]) / 7            (symmetric, no zero-point)
q[m, k]  = clamp(round(x[m, k] / scale[m]), -8, 7)
```

Two nibbles per byte. Nibble 0 holds element `2k`, nibble 1 holds element
`2k+1`. Per-backend layout details may diverge — see each impl.

## Public header
`include/quantize_act_int4.h`

## Backend notes
- **cuda/** — warp-reduce amax → scale → quantize-and-pack in one pass.
- **ascend/** — vector-unit reduction; pack on-chip.
