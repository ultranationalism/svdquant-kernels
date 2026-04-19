"""Triton kernel(s) for `quantize_w4a4_act_fuse_lora`.

The full fused op (quantize + LoRA-down in one pass) is not yet
implemented — see `README.md` for the contract and the nunchaku
reference at `gemm_w4a4.cuh:1096-1184`.

For now this module exposes `lora_down`, the isolated LoRA-down
matmul (`input @ lora_down` → fp32). Splitting it off as a standalone
prototype is fine because LoRA-down is the least performance-critical
piece of the op (small K→R, AI well below the HBM ridge even in
isolation) and because a standalone Triton matmul runs on any Triton
target unchanged, which makes it a useful scaffold for wiring up the
later fused version on both CUDA and `triton-ascend`.
"""
from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _lora_down_kernel(
    input_ptr, lora_down_ptr, out_ptr,
    M, N, R,
    stride_im, stride_in,
    stride_ln, stride_lr,
    stride_om, stride_or,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_r = tl.arange(0, BLOCK_R)

    mask_m = offs_m < M
    mask_r = offs_r < R

    acc = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)

    input_base = input_ptr + offs_m[:, None] * stride_im
    lora_base = lora_down_ptr + offs_r[None, :] * stride_lr

    for k in range(0, N, BLOCK_N):
        k_offs = k + offs_n
        mask_n = k_offs < N

        a = tl.load(
            input_base + k_offs[None, :] * stride_in,
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        )
        b = tl.load(
            lora_base + k_offs[:, None] * stride_ln,
            mask=mask_n[:, None] & mask_r[None, :],
            other=0.0,
        )
        acc += tl.dot(a, b, out_dtype=tl.float32)

    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_r[None, :] * stride_or,
        acc,
        mask=mask_m[:, None] & mask_r[None, :],
    )


def lora_down(
    input: torch.Tensor,       # [M, N] fp16/bf16
    weight: torch.Tensor,      # [N, R] fp16/bf16, same dtype as input
    *,
    block_m: int = 64,
    block_n: int = 64,
) -> torch.Tensor:             # [M, R] fp32
    """Standalone LoRA-down projection. Prototype for the fused op."""
    assert input.dim() == 2 and weight.dim() == 2
    assert input.is_cuda and weight.is_cuda
    assert input.dtype == weight.dtype
    assert input.dtype in (torch.float16, torch.bfloat16)

    M, N = input.shape
    N2, R = weight.shape
    assert N == N2, f"K mismatch: input K={N}, weight K={N2}"

    out = torch.empty((M, R), dtype=torch.float32, device=input.device)

    # R is a small LoRA rank (32/128/256 in ZImage Turbo); round to next
    # power of two so the whole R axis fits in one block, and enforce the
    # 16-min dim that `tl.dot` needs on sm_80+.
    block_r = max(16, triton.next_power_of_2(R))

    grid = (triton.cdiv(M, block_m),)
    _lora_down_kernel[grid](
        input, weight, out,
        M, N, R,
        input.stride(0), input.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_R=block_r,
    )
    return out


def quantize_w4a4_act_fuse_lora(
    input: torch.Tensor,             # [M, N]   fp16/bf16
    lora_down: torch.Tensor,         # [N, R]   fp16/bf16 (smooth pre-absorbed offline)
    smooth: Optional[torch.Tensor],  # [N]      fp16/bf16 or None
    *,
    fp4: bool,                       # True = NVFP4 (CUDA), False = INT4 (Ascend)
    pad_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full fused op — not yet implemented. See README.md.

    Output format picks per backend: NVFP4 on CUDA (B200 tcgen05 has
    no INT4 scaled-MMA), INT4 on Ascend (its cube unit has no FP4).
    Caller passes `fp4` explicitly — we don't auto-detect from the
    tensor's device because triton-ascend may or may not show up as
    a distinct device type depending on the install.

    We don't support nunchaku's `fuse_glu` path: that would require
    vLLM to hand us the pre-GLU [M, 2N] tensor, which is a pipeline
    intrusion we're avoiding.
    """
    raise NotImplementedError(
        "Triton kernel body pending. See README.md for the contract "
        "and `tmp/nunchaku/src/kernels/zgemm/gemm_w4a4.cuh:1096` for "
        "the CUDA reference. The LoRA-down piece is available in "
        "isolation as `lora_down(input, weight)`."
    )
