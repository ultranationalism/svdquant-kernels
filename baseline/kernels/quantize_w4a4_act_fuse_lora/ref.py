"""Pure-PyTorch reference for `quantize_w4a4_act_fuse_lora` (NVFP4 path).

Produces the same three outputs as nunchaku's fused kernel:
    - qout        [M_pad, K/2]      uint8       two NVFP4 nibbles per byte
    - oscales     [K/16,  M_pad]    fp8_e4m3fn  per-16-K-block scale
    - lora_act    [M_pad, R]        fp32

Readability over speed. No fused ops, no torch.compile.
NVFP4 math lives in `.._nvfp4`; this ref just adds the LoRA-down
projection and the optional smooth-divide.
"""
from __future__ import annotations

import torch

from .._nvfp4 import quantize_nvfp4_rows


def quantize_w4a4_act_fuse_lora_ref(
    input: torch.Tensor,            # [M, K] fp16/bf16
    lora_down: torch.Tensor,        # [K, R] fp16/bf16
    smooth: torch.Tensor | None,    # [K]    fp16/bf16 or None
    *,
    pad_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """NVFP4 reference. Shapes in module docstring."""
    assert input.dtype in (torch.float16, torch.bfloat16)
    assert lora_down.dtype == input.dtype
    assert input.dim() == 2 and lora_down.dim() == 2
    M, K = input.shape
    K2, R = lora_down.shape
    assert K == K2

    M_pad = ((M + pad_size - 1) // pad_size) * pad_size

    # --- LoRA-down: unsmoothed X @ lora_down ---
    # Paper says X̂ @ L2^T; nunchaku absorbs diag(1/smooth) into stored
    # lora_down offline, so runtime uses raw X. The calibration pipeline
    # is the authority on what's in `lora_down` — we just take it as given.
    lora_act = input.to(torch.float32) @ lora_down.to(torch.float32)
    lora_act_pad = torch.zeros(M_pad, R, dtype=torch.float32, device=input.device)
    lora_act_pad[:M] = lora_act

    # --- Smooth divide + NVFP4 quantize (delegated) ---
    x = input.to(torch.float32)
    if smooth is not None:
        x = x / smooth.to(torch.float32)
    qout, oscales = quantize_nvfp4_rows(x, pad_size=pad_size)

    return qout, oscales, lora_act_pad
