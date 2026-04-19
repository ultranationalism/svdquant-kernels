"""Shared harness for `gemm_w4a4` smoke + bench.

Both `tmp/smoke_gemm.py` and `tmp/bench_gemm.py` import from here, so
the shape set and the seed → input mapping are defined **in exactly
one place**. Reproducibility across machines (local SM_120, Modal
B200) comes from the CPU-side RNG in `make_gemm_inputs`: given the
same seed, every hardware generates bit-identical inputs.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

# (M, K, N, R). Mixed to cover: small-batch sanity, full-batch, FFN up
# (wide N), FFN down (wide K), LoRA ranks at both ends.
GEMM_SHAPES: tuple[tuple[int, int, int, int], ...] = (
    ( 256,  3840,  3072, 128),
    (4352,  3840,  3072, 128),
    (4352,  3840, 15360, 128),
    (4352, 15360,  3840, 128),
    (4352, 10240,  3072,  32),
    (4352, 10240,  3072, 256),
)

GEMM_DTYPES: tuple[torch.dtype, ...] = (torch.float16, torch.bfloat16)


@dataclass(frozen=True)
class GemmInputs:
    """Seeded inputs for one `gemm_w4a4` test case, all on `device`."""
    x: torch.Tensor             # [M, K]  dtype — unquantized activation
    w: torch.Tensor             # [N, K]  dtype — unquantized weight
    lora_down: torch.Tensor     # [K, R]  dtype
    lora_up: torch.Tensor       # [N, R]  dtype
    smooth: torch.Tensor        # [K]     dtype
    wcscales: torch.Tensor      # [N]     dtype
    bias: torch.Tensor          # [N]     dtype
    smooth_next: torch.Tensor   # [N]     dtype — enables next-layer quant


def _seeded(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    """Deterministic fp32 CPU draw — bit-exact across any hardware."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float32)


def make_gemm_inputs(
    M: int, K: int, N: int, R: int,
    dtype: torch.dtype,
    *,
    seed: int = 0,
    device: str | torch.device = "cuda",
) -> GemmInputs:
    """CPU-side seeded → target-dtype CUDA inputs for one shape.

    `seed` offsets the per-tensor seed deterministically, so each
    (M, K, N, R, dtype) call produces the same bits regardless of which
    hardware generates them. The weight/lora/affine scales are kept in
    magnitudes where quantization stays in-range (no saturation at the
    fp4_max = 6 or fp8_e4m3_max = 448 boundaries).
    """
    x_fp32           = _seeded((M, K),   seed + 0)
    w_fp32           = _seeded((N, K),   seed + 1) * 0.1
    lora_down_fp32   = _seeded((K, R),   seed + 2) * 0.1
    lora_up_fp32     = _seeded((N, R),   seed + 3) * 0.1
    smooth_fp32      = _seeded((K,),     seed + 4).abs() + 0.5
    wcscales_fp32    = _seeded((N,),     seed + 5).abs() + 0.5
    bias_fp32        = _seeded((N,),     seed + 6)
    smooth_next_fp32 = _seeded((N,),     seed + 7).abs() + 0.5

    def _cast(t: torch.Tensor) -> torch.Tensor:
        return t.to(dtype).to(device)

    return GemmInputs(
        x=_cast(x_fp32),
        w=_cast(w_fp32),
        lora_down=_cast(lora_down_fp32),
        lora_up=_cast(lora_up_fp32),
        smooth=_cast(smooth_fp32),
        wcscales=_cast(wcscales_fp32),
        bias=_cast(bias_fp32),
        smooth_next=_cast(smooth_next_fp32),
    )
