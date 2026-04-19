# Each subdir is a "kernel pod": self-contained sources for one SVDQuant op,
# with `include/` (public header), `cuda/kernel.cu`, and `ascend/kernel.cpp`.
# No per-pod CMakeLists — the helper below handles them uniformly.

function(svdquant_add_kernel_pod name)
    set(_root "${CMAKE_CURRENT_LIST_DIR}/${name}")
    set(_sources "")
    if(SVDQUANT_ENABLE_CUDA AND EXISTS "${_root}/cuda/kernel.cu")
        list(APPEND _sources "${_root}/cuda/kernel.cu")
    endif()
    if(SVDQUANT_ENABLE_ASCEND AND EXISTS "${_root}/ascend/kernel.cpp")
        list(APPEND _sources "${_root}/ascend/kernel.cpp")
    endif()
    if(NOT _sources)
        message(STATUS "svdquant: skipping kernel pod '${name}' (no enabled sources)")
        return()
    endif()

    add_library(svdquant_${name} OBJECT ${_sources})
    target_link_libraries(svdquant_${name} PUBLIC svdquant::common)
    target_include_directories(svdquant_${name} PUBLIC "${_root}/include")
    if(SVDQUANT_ENABLE_ASCEND)
        target_include_directories(svdquant_${name} PRIVATE ${CANN_INCLUDE_DIRS})
    endif()
    add_library(svdquant::${name} ALIAS svdquant_${name})
endfunction()

svdquant_add_kernel_pod(gemm_w4a4)

# Preprocessing op `quantize_w4a4_act_fuse_lora` is memory-bound and
# needs to run on both CUDA and Ascend, so it lives in the Triton
# directory (`triton_kernels/`) instead of here — Triton's Python +
# LLVM toolchain isn't driven by CMake. See `triton_kernels/README.md`
# for the library-choice rationale.
