"""Pure-PyTorch reference for `gemm_w4a4` (NVFP4 path).

Eats the same canonical NVFP4 layout that
`baseline/kernels/quantize_w4a4_act_fuse_lora/ref.py` emits:

    act          [M_pad, K/2]   uint8       two NVFP4 nibbles per byte
    wgt          [N,     K/2]   uint8
    ascales      [K/16,  M_pad] fp8_e4m3fn  per-16-K-block act scale
    wscales      [K/16,  N]     fp8_e4m3fn  per-16-K-block weight scale
    lora_act_in  [M_pad, R]     fp32        = previous-op output
    lora_up      [N,     R]     fp16/bf16
    bias         [N]            fp16/bf16   (optional)
    wcscales     [N]            fp16/bf16   (optional per-channel scale)
    smooth_next  [N]            fp16/bf16   (optional; enables next-layer quant)

Output:
    out          [M_pad, N]     lora_up.dtype
    qout         [M_pad, N/2]   uint8       (None if smooth_next is None)
    oscales      [N/16,  M_pad] fp8_e4m3fn  (None if smooth_next is None)

All math in fp32; `out` is cast at the end. Readability over speed —
no fused ops, no torch.compile. Counterpart to nunchaku's host launcher
at `tmp/nunchaku/src/kernels/zgemm/gemm_w4a4.cu:34-105` (linear-only
subset; attention-fusion params intentionally omitted).
"""
from __future__ import annotations

import torch

from .._nvfp4 import dequantize_nvfp4_rows, quantize_nvfp4_rows


def gemm_w4a4_ref(
    act: torch.Tensor,
    wgt: torch.Tensor,
    ascales: torch.Tensor,
    wscales: torch.Tensor,
    lora_act_in: torch.Tensor,
    lora_up: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    wcscales: torch.Tensor | None = None,
    alpha: float = 1.0,
    smooth_next: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """NVFP4 reference. See module docstring for shapes."""
    assert act.dtype == torch.uint8 and wgt.dtype == torch.uint8
    assert ascales.dtype == torch.float8_e4m3fn
    assert wscales.dtype == torch.float8_e4m3fn
    M_pad, K2 = act.shape
    N, K2w = wgt.shape
    assert K2 == K2w, f"act/wgt K disagree: {K2} vs {K2w}"
    K = K2 * 2
    assert K % 16 == 0, "NVFP4 group size is 16"
    R = lora_act_in.shape[1]
    assert lora_act_in.shape == (M_pad, R)
    assert lora_up.shape == (N, R)
    assert ascales.shape == (K // 16, M_pad)
    assert wscales.shape == (K // 16, N)

    # --- Main matmul in fp32 ---
    act_fp32 = dequantize_nvfp4_rows(act, ascales)    # [M_pad, K]
    wgt_fp32 = dequantize_nvfp4_rows(wgt, wscales)    # [N,     K]
    y = (act_fp32 @ wgt_fp32.T) * alpha               # [M_pad, N]

    # --- LoRA-up residual ---
    y = y + lora_act_in.to(torch.float32) @ lora_up.to(torch.float32).T

    # --- per-channel affine ---
    if wcscales is not None:
        assert wcscales.shape == (N,)
        y = y * wcscales.to(torch.float32)
    if bias is not None:
        assert bias.shape == (N,)
        y = y + bias.to(torch.float32)

    out_dtype = lora_up.dtype
    y_out = y.to(out_dtype)

    # --- optional next-layer NVFP4 quantize ---
    qout: torch.Tensor | None = None
    oscales: torch.Tensor | None = None
    if smooth_next is not None:
        assert smooth_next.shape == (N,)
        assert N % 16 == 0, "NVFP4 group size is 16"
        y_for_quant = y / smooth_next.to(torch.float32)
        qout, oscales = quantize_nvfp4_rows(y_for_quant, pad_size=1)

    return y_out, qout, oscales
