"""NVFP4 row-wise quantization primitive.

NVFP4 = FP4 values (E2M1: sign + 3-bit exponent+mantissa encoding a
level from `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`) with per-16-K-block
FP8-E4M3 scales. NVIDIA-published group convention.

`quantize_nvfp4_rows` / `dequantize_nvfp4_rows` are the canonical pair
used by both `quantize_w4a4_act_fuse_lora` and `gemm_w4a4` refs — and
by the golden-dump generator when it emits random weights for
kernel benchmarks.
"""
from __future__ import annotations

import torch

# Positive magnitudes of the NVFP4 E2M1 value set. Encoding of a nibble
# byte = index into this table (low 3 bits), sign bit in position 3.
_E2M1_LEVELS = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)
NVFP4_AMAX = 6.0
# FP8-E4M3 max finite value. Nunchaku clamps scales to this pre-cast
# (`gemm_w4a4.cuh:93`); mirror for bit-for-bit agreement.
FP8_E4M3_MAX = 448.0


def _quantize_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round fp32 values in `[-6, 6]` to nearest NVFP4 level, return uint8 nibble.

    Midpoint-then-floor rounding (matches hardware `cvt.rn` on the open
    intervals). Tie behaviour may differ on the few rational midpoints
    (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0).
    """
    abs_x = x.abs().clamp_max(NVFP4_AMAX)
    thresholds = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        dtype=torch.float32, device=x.device,
    )
    idx = (abs_x.unsqueeze(-1) >= thresholds).sum(-1)
    sign_bit = (x < 0).to(torch.uint8) << 3
    return idx.to(torch.uint8) | sign_bit


def _pack_nibbles(nibs: torch.Tensor) -> torch.Tensor:
    """Pack last-dim pairs of 4-bit nibbles into uint8. Low nibble = even k."""
    assert nibs.shape[-1] % 2 == 0
    lo = nibs[..., 0::2]
    hi = nibs[..., 1::2]
    return (lo | (hi << 4)).to(torch.uint8)


def _unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """`[*, K/2]` uint8 → `[*, K]` uint8 nibbles. Inverse of `_pack_nibbles`."""
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    out = torch.stack([lo, hi], dim=-1)
    return out.view(*packed.shape[:-1], packed.shape[-1] * 2)


def quantize_nvfp4_rows(
    x: torch.Tensor,
    *,
    pad_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-wise NVFP4 quantization.

    Args:
        x:        `[rows, K]` floating-point tensor. fp16/bf16/fp32 all OK.
        pad_size: row-dim alignment (activations use 256; weights use 1).

    Returns:
        packed: `[rows_pad, K/2]`  uint8 — NVFP4 nibbles, low nibble = even k.
        scales: `[K/16, rows_pad]` fp8_e4m3fn — nunchaku-style transposed layout.

    Padding rows are zeros (packed) and the FP8 cast of 1e-12 (scales).
    """
    assert x.dim() == 2
    rows, K = x.shape
    assert K % 16 == 0, "NVFP4 group size is 16"
    rows_pad = ((rows + pad_size - 1) // pad_size) * pad_size

    x_pad = torch.zeros(rows_pad, K, dtype=torch.float32, device=x.device)
    x_pad[:rows] = x.to(torch.float32)

    blocks = x_pad.view(rows_pad, K // 16, 16)
    amax = blocks.abs().amax(dim=-1)                                 # [rows_pad, K/16]
    scale_f32 = (amax / NVFP4_AMAX).clamp(min=1e-12, max=FP8_E4M3_MAX)
    scale_fp8 = scale_f32.to(torch.float8_e4m3fn)
    scale_back = scale_fp8.to(torch.float32)

    x_scaled = blocks / scale_back.unsqueeze(-1)                     # [rows_pad, K/16, 16]
    nibs = _quantize_e2m1(x_scaled).view(rows_pad, K)
    packed = _pack_nibbles(nibs)                                     # [rows_pad, K/2]
    scales = scale_fp8.transpose(0, 1).contiguous()                  # [K/16, rows_pad]
    return packed, scales


def dequantize_nvfp4_rows(
    packed: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Inverse of `quantize_nvfp4_rows`. Returns fp32 `[rows, K]`."""
    rows, K2 = packed.shape
    K = K2 * 2
    assert scales.shape == (K // 16, rows), (
        f"scales shape {tuple(scales.shape)} != expected ({K // 16}, {rows})"
    )
    levels = _E2M1_LEVELS.to(packed.device)
    nibs = _unpack_nibbles(packed).view(rows, K)
    mag = (nibs & 0x07).long()
    sign = 1.0 - ((nibs >> 3) & 0x01).float() * 2.0
    vals = levels[mag] * sign                                        # [rows, K] fp32
    scale_per_block = scales.transpose(0, 1).to(torch.float32)       # [rows, K/16]
    return (vals.view(rows, K // 16, 16) * scale_per_block.unsqueeze(-1)).view(rows, K)
