cmake_minimum_required(VERSION 3.22)

project(svdquant_kernels
    VERSION 0.0.0
    DESCRIPTION "Cross-architecture kernels for SVDQuant (W4A4 + low-rank correction)"
    LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# ---- Backend toggles -------------------------------------------------------
option(SVDQUANT_ENABLE_CUDA   "Build CUDA (NVIDIA GPU) backend" ON)
option(SVDQUANT_ENABLE_ASCEND "Build Ascend NPU backend"        ON)
option(SVDQUANT_BUILD_TESTS      "Build unit / numeric tests"   OFF)
option(SVDQUANT_BUILD_BENCHMARKS "Build micro-benchmarks"       OFF)

# ---- CUDA probe ------------------------------------------------------------
if(SVDQUANT_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        include(cuda_arch)
        message(STATUS "svdquant: CUDA enabled (archs: ${SVDQUANT_CUDA_ARCHS})")
    else()
        message(WARNING "svdquant: CUDA requested but no CUDA compiler found — disabling")
        set(SVDQUANT_ENABLE_CUDA OFF)
    endif()
endif()

# ---- Ascend probe ----------------------------------------------------------
if(SVDQUANT_ENABLE_ASCEND)
    find_package(CANN QUIET)
    if(CANN_FOUND)
        message(STATUS "svdquant: Ascend enabled (CANN: ${CANN_ROOT})")
    else()
        message(WARNING "svdquant: Ascend requested but CANN not found — disabling (set ASCEND_HOME_PATH)")
        set(SVDQUANT_ENABLE_ASCEND OFF)
    endif()
endif()

if(NOT SVDQUANT_ENABLE_CUDA AND NOT SVDQUANT_ENABLE_ASCEND)
    message(FATAL_ERROR "svdquant: no backend enabled — nothing to build")
endif()

# ---- Subdirs ---------------------------------------------------------------
add_subdirectory(csrc)

if(SVDQUANT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(SVDQUANT_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
