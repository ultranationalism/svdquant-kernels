# SVDQuant CUDA target SM list.
# Scope is SM_100 and SM_103 (data-center Blackwell) only. The W4A4
# path relies on Blackwell-specific tensor-memory ops (`tcgen05` 5th-gen
# tensor cores with block-scaled MMA). Earlier archs (Turing through
# Hopper) and consumer Blackwell (SM_120a / SM_121a) are covered by
# `nunchaku` — see `tmp/nunchaku/setup.py:41-64` — so they are
# deliberately omitted rather than silently producing a slow fallback.
if(NOT DEFINED SVDQUANT_CUDA_ARCHS)
    set(SVDQUANT_CUDA_ARCHS "100;103" CACHE STRING "CUDA SM archs to build")
endif()

set(CMAKE_CUDA_ARCHITECTURES ${SVDQUANT_CUDA_ARCHS})
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Lets per-file code gate on specific arches, e.g. #ifdef SVDQUANT_HAS_SM90.
foreach(_arch IN LISTS SVDQUANT_CUDA_ARCHS)
    add_compile_definitions(SVDQUANT_HAS_SM${_arch}=1)
endforeach()
