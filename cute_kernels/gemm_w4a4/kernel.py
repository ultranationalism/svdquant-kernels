"""gemm_w4a4 CuTe DSL kernel (Blackwell SM_100/SM_103).

Host-facing contract (v0 — main NVFP4 only; LoRA / wcscales / bias /
next-layer quant land in v1–v3):

    launch(act, wgt, ascales, wscales, *, out_dtype=torch.float16) -> out

Inputs (NVFP4-packed, as produced by
`triton_kernels/quantize_w4a4_act_fuse_lora/`):

    act       [M, K // 2]    uint8           two E2M1 nibbles per byte
    wgt       [N, K // 2]    uint8           two E2M1 nibbles per byte
    ascales   [K // 16, M]   fp8_e4m3fn      per-16-K-block act scale
    wscales   [K // 16, N]   fp8_e4m3fn      per-16-K-block wgt scale

Output:
    out       [M, N]         out_dtype (fp16 or bf16)

Reference skeleton: `tmp/cutlass/examples/python/CuTeDSL/blackwell/
dense_blockscaled_gemm_persistent.py` (stable API). The cleaner
`experimental/blackwell/dense_block_scaled_gemm.py` uses
`cute.experimental.*` which `cutlass-dsl` hard-gates to CUDA toolkit
13.1+; our runtime is CUDA 13.0, so we're on the stable path.

Design doc: `docs/kernels/gemm_w4a4.md`.
"""
from __future__ import annotations

import os
from typing import Tuple, Type

import torch

# Arch gate: probe returns SM_100/SM_103 on B200/B30x, SM_120 locally
# (consumer Blackwell — rejects tcgen05 NVFP4 atoms). Allow the env to
# force an arch for trace-level checks on a wrong-arch box.
os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")

import cutlass                                           # noqa: E402
import cutlass.cute as cute                              # noqa: E402
import cutlass.torch as cutlass_torch                    # noqa: E402
import cutlass.utils as utils                            # noqa: E402
from cutlass.cute.nvgpu import tcgen05                   # noqa: E402
from cutlass.cute.runtime import make_ptr                # noqa: E402

# --- compile cache ---------------------------------------------------------
# `cute.compile` is expensive (MLIR lowering + PTX emit). One compiled
# kernel per (a_dtype, b_dtype, sf_dtype, c_dtype, tiler) combo; shapes
# (M, N, K) are runtime args.
_COMPILED_CACHE: dict[tuple, object] = {}

# --- config ---------------------------------------------------------------
# v0 default: 1SM 128×256 (single-CTA instruction). 2SM 256×256 is the
# eventual production default per design §6; starting at 1SM keeps the
# first implementation simpler (no cluster-aware barriers), and the
# M=256 shape in the smoke set prefers 1SM anyway.
_MMA_INST_MN: Tuple[int, int] = (128, 256)
_SF_VEC_SIZE: int = 16                 # NVFP4 block size
_NUM_MAIN_STAGES: int = 4              # TMA → MMA pipeline depth
_NUM_ACC_STAGES: int = 1


def _check_inputs(
    act: torch.Tensor,
    wgt: torch.Tensor,
    ascales: torch.Tensor,
    wscales: torch.Tensor,
    out_dtype: torch.dtype,
) -> Tuple[int, int, int]:
    """Validate layout + dtype. Returns (M, N, K)."""
    assert act.is_cuda and wgt.is_cuda and ascales.is_cuda and wscales.is_cuda, \
        "all inputs must live on CUDA"
    assert act.dtype == torch.uint8 and wgt.dtype == torch.uint8, \
        f"act/wgt must be uint8 NVFP4-packed; got {act.dtype}/{wgt.dtype}"
    assert ascales.dtype == torch.float8_e4m3fn and wscales.dtype == torch.float8_e4m3fn, \
        f"scales must be fp8_e4m3fn; got {ascales.dtype}/{wscales.dtype}"
    assert out_dtype in (torch.float16, torch.bfloat16), \
        f"out_dtype must be fp16 or bf16; got {out_dtype}"

    M, K_half = act.shape
    N, K_half_w = wgt.shape
    assert K_half == K_half_w, f"act/wgt K disagree: {K_half} vs {K_half_w}"
    K = K_half * 2
    assert K % _SF_VEC_SIZE == 0, f"K ({K}) must be a multiple of {_SF_VEC_SIZE}"

    K_groups = K // _SF_VEC_SIZE
    assert ascales.shape == (K_groups, M), \
        f"ascales shape {tuple(ascales.shape)} != ({K_groups}, {M})"
    assert wscales.shape == (K_groups, N), \
        f"wscales shape {tuple(wscales.shape)} != ({K_groups}, {N})"

    # 2CTA / NVFP4 atom alignment
    assert M % _MMA_INST_MN[0] == 0, f"M ({M}) must be a multiple of {_MMA_INST_MN[0]}"
    assert N % _MMA_INST_MN[1] == 0, f"N ({N}) must be a multiple of {_MMA_INST_MN[1]}"
    assert K % 32 == 0, f"K ({K}) must be a multiple of 32 for NVFP4 alignment"
    return M, N, K


def _torch_to_fp4_ptr(packed_uint8: torch.Tensor) -> "cute.Pointer":
    """uint8 tensor of shape [., K/2] → cute.Pointer typed Float4E2M1FN
    addressing the same bytes. No copy."""
    return make_ptr(
        cutlass.Float4E2M1FN,
        packed_uint8.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=32,       # 32-elem (16-byte) alignment for FP4 TMA
    )


def _torch_to_fp8sf_ptr(sf_uint8: torch.Tensor) -> "cute.Pointer":
    return make_ptr(
        cutlass.Float8E4M3FN,
        sf_uint8.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )


def _torch_to_c_ptr(out: torch.Tensor) -> "cute.Pointer":
    dtype = {
        torch.float16:  cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
    }[out.dtype]
    return make_ptr(dtype, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)


def _cutlass_out_dtype(out_dtype: torch.dtype) -> Type[cutlass.Numeric]:
    return {torch.float16: cutlass.Float16, torch.bfloat16: cutlass.BFloat16}[out_dtype]


def launch(
    act: torch.Tensor,
    wgt: torch.Tensor,
    ascales: torch.Tensor,
    wscales: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """v0: y = scaled_mma(act_nvfp4, wgt_nvfp4) — no LoRA, no affine."""
    M, N, K = _check_inputs(act, wgt, ascales, wscales, out_dtype)

    out = torch.empty((M, N), dtype=out_dtype, device=act.device)

    cache_key = (
        cutlass.Float4E2M1FN, cutlass.Float4E2M1FN,
        cutlass.Float8E4M3FN, _cutlass_out_dtype(out_dtype),
        _MMA_INST_MN, _SF_VEC_SIZE, _NUM_MAIN_STAGES, _NUM_ACC_STAGES,
    )
    compiled = _COMPILED_CACHE.get(cache_key)
    if compiled is None:
        compiled = _compile_v0(
            ab_dtype=cutlass.Float4E2M1FN,
            sf_dtype=cutlass.Float8E4M3FN,
            c_dtype=_cutlass_out_dtype(out_dtype),
        )
        _COMPILED_CACHE[cache_key] = compiled

    stream = torch.cuda.current_stream().cuda_stream
    compiled(
        _torch_to_fp4_ptr(act),
        _torch_to_fp4_ptr(wgt),
        _torch_to_fp8sf_ptr(ascales),
        _torch_to_fp8sf_ptr(wscales),
        _torch_to_c_ptr(out),
        (cutlass.Int32(M), cutlass.Int32(N), cutlass.Int32(K), cutlass.Int32(1)),
        stream,
    )
    return out


# --- device kernel --------------------------------------------------------


def _compile_v0(
    *,
    ab_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
):
    """AOT-compile the v0 scaled-MMA kernel and return the callable."""
    kernel_obj = Sm100GemmW4A4V0(
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        mma_inst_mn=_MMA_INST_MN,
        sf_vec_size=_SF_VEC_SIZE,
        num_main_stages=_NUM_MAIN_STAGES,
        num_acc_stages=_NUM_ACC_STAGES,
    )
    a_ptr = make_ptr(ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=32)
    b_ptr = make_ptr(ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=32)
    sfa_ptr = make_ptr(sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    sfb_ptr = make_ptr(sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    c_ptr = make_ptr(c_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)

    # Dummy stream for AOT compile; real stream is passed at call time.
    import cuda.bindings.driver as cuda_drv
    dummy_stream = cuda_drv.CUstream(0)

    return cute.compile(
        kernel_obj,
        a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr,
        (cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0)),
        dummy_stream,
    )


class Sm100GemmW4A4V0:
    """v0 device-side kernel: main NVFP4 scaled-MMA only.

    Not persistent; one CTA per output tile. Warp-spec split is:

        warp 0       — TMA store driver
        warps 0–3    — epilogue (tmem → rmem → smem → gmem)
        warp 4       — MMA issue
        warp 5       — TMA load

    Total 6 warps, 192 threads per CTA (same layout as the experimental
    1021-line example — we re-derive it on stable-API primitives).

    The NVFP4 scaled-MMA atom, SF TMA path, and K-loop issue pattern
    are ported from `dense_blockscaled_gemm_persistent.py`, minus the
    persistent tile scheduler. See the TODO block in `kernel()` for
    the exact porting map.
    """

    def __init__(
        self,
        *,
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        mma_inst_mn: Tuple[int, int],
        sf_vec_size: int,
        num_main_stages: int,
        num_acc_stages: int,
    ):
        self.ab_dtype = ab_dtype
        self.sf_dtype = sf_dtype
        self.c_dtype = c_dtype
        self.acc_dtype = cutlass.Float32
        self.mma_inst_shape_mn = mma_inst_mn
        self.sf_vec_size = sf_vec_size
        self.num_main_stages = num_main_stages
        self.num_acc_stages = num_acc_stages

        # 1SM single-CTA atom for v0. 2SM upgrade is a later tuning pass.
        self.use_2cta_instrs = False
        self.cta_group = tcgen05.CtaGroup.ONE
        self.tma_store_stages = 4

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        problem_mnkl: Tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32, cutlass.Int32],
        stream,
    ):
        m, n, k, l = problem_mnkl

        # A : [M, K] E2M1   K-major   (underlying bytes: uint8 [M, K/2])
        # B : [N, K] E2M1   K-major
        # C : [M, N] out_dtype  N-major (row-major)
        a_layout = cute.make_ordered_layout(
            (cute.assume(m, 32), k, l), order=(1, 0, 2)
        )
        b_layout = cute.make_ordered_layout(
            (cute.assume(n, 32), k, l), order=(1, 0, 2)
        )
        c_layout = cute.make_ordered_layout(
            (m, cute.assume(n, 32), l), order=(1, 0, 2)
        )
        a_tensor = cute.make_tensor(a_ptr, a_layout)
        b_tensor = cute.make_tensor(b_ptr, b_layout)
        c_tensor = cute.make_tensor(c_ptr, c_layout)  # noqa: F841

        # SFA/SFB tensors derive from A/B shape via the block-scale atom
        # layout (see blockscaled_utils.tile_atom_to_shape_SF).
        import cutlass.utils.blockscaled_layout as blockscaled_utils
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            a_tensor.shape, self.sf_vec_size
        )
        sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)  # noqa: F841
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b_tensor.shape, self.sf_vec_size
        )
        sfb_tensor = cute.make_tensor(sfb_ptr, sfb_layout)  # noqa: F841

        import cutlass.utils.blackwell_helpers as sm100_utils
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.ab_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )  # noqa: F841

        # ============================================================
        # TODO(v0): port the following from dense_blockscaled_gemm_persistent.py
        # (stable API). Concrete line refs in that file:
        #
        #   1. Grid + launch math + SharedStorage struct        L ~520–680
        #   2. SMEM layouts for A / B / SFA / SFB               L ~505–560
        #   3. TMA atoms for A / B / SFA / SFB (non-clustered)  L ~501–610
        #   4. TMEM accumulator layout + allocator              L ~620–670
        #   5. Mainloop TMA→MMA pipeline (`cutlass.pipeline`)    L ~700–790
        #   6. Warp-spec split (load / mma / epilogue)           L ~800–900
        #   7. MMA K-loop using `cute.gemm(tiled_mma, ...)` +
        #      `make_blockscaled_tiled_copy` for SF SMEM→TMEM    L ~950–1100
        #   8. Epilogue TMEM→RMEM→SMEM→TMA store                 L ~1800–2100
        #
        # Drops from persistent.py (not needed for v0 / any v):
        #   - TileScheduler + persistent CTA loop  (whole `schedule`)
        #   - cluster shapes > 1  (single-CTA fine for first pass)
        #   - dynamic cluster sizes / max_active_clusters
        #   - overlapping_accum (no acc-stage pressure at 1 stage)
        #
        # Once this body compiles + passes `tmp/smoke_gemm.py` on Modal,
        # v1 adds LoRA MMA issue inside the K-loop per design §2.
        # ============================================================
        raise NotImplementedError(
            "gemm_w4a4 v0 device body lands in a follow-up turn; "
            "host plumbing + config + tensor-layout setup are in place."
        )
