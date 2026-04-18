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

svdquant_add_kernel_pod(w4a4_gemm)
svdquant_add_kernel_pod(lowrank_branch)
svdquant_add_kernel_pod(fused_svdquant_linear)
svdquant_add_kernel_pod(quantize_act_int4)
svdquant_add_kernel_pod(pack_weight_int4)
