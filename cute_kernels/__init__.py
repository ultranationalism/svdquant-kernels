"""CUDA CuTe DSL kernels for SVDQuant (Blackwell SM_100/SM_103).

One directory per op; each pod is a `@cute.jit` kernel + torch-tensor
host wrapper, JIT-lowered through MLIR → PTX by `cutlass-dsl` at first
call. See CLAUDE.md for the three-flavor split (CuTe DSL vs Triton vs
AscendC).
"""
