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
# SOC_VERSION picks the AICore arch ccec targets via ascendc.cmake.
# ascend910b1 = A2 (910B). Override on a 910C / Atlas A3 host.
set(SVDQUANT_SOC_VERSION "ascend910b1" CACHE STRING "Ascend SoC version for ccec target")

if(SVDQUANT_ENABLE_ASCEND)
    find_package(CANN QUIET)
    if(CANN_FOUND)
        message(STATUS "svdquant: Ascend enabled (CANN: ${CANN_ROOT})")
        if(CANN_ASCENDC_CMAKE)
            # ascendc.cmake exposes ascendc_library() for ccec cross-compile.
            # Both vars must be set *before* including (they're read at
            # include time, not lazily). ASCEND_CANN_PACKAGE_PATH is what
            # host_config.cmake checks first; CANN_ROOT is the resolved
            # path from FindCANN.
            set(SOC_VERSION "${SVDQUANT_SOC_VERSION}")
            set(ASCEND_CANN_PACKAGE_PATH "${CANN_ROOT}")
            # Override the Python interpreter ascendc.cmake feeds to
            # extract_host_stub.py. The wrapper normpath's /./ segments
            # in argv before exec'ing real python3 — see the script for
            # the full context. CACHE overrides what host_config.cmake
            # would otherwise set to plain "python3".
            set(ASCEND_PYTHON_EXECUTABLE "${CMAKE_SOURCE_DIR}/scripts/ascendc_python_wrapper.py"
                CACHE STRING "python executable program (svdquant wrapper)" FORCE)
            # Skip the POST_BUILD recompile_binary.py step. It expects
            # CMake to have produced a `link.txt` for the device target,
            # which CMake only emits for SHARED/MODULE/EXECUTABLE — STATIC
            # libs (which is what ascendc_library produces for us) get
            # only an `ar`-based archive command, no link.txt. The
            # recompile step would relink the host-stub OBJECT plus the
            # device .o into a single archive; we don't need that because
            # our host launcher invokes the device kernel directly via
            # the AscendC runtime API, not through the merged stub.
            set(ASCEND_KERNEL_LAUNCH_ONLY ON CACHE BOOL "Skip ascendc host-stub merge step" FORCE)
            include("${CANN_ASCENDC_CMAKE}")
            message(STATUS "svdquant: ascendc.cmake loaded (SOC_VERSION=${SOC_VERSION})")
            set(SVDQUANT_ASCENDC_AVAILABLE ON)
        else()
            message(STATUS "svdquant: ascendc.cmake not found — device-side __aicore__ "
                           "sources will be skipped; only host-side .cpp will compile")
            set(SVDQUANT_ASCENDC_AVAILABLE OFF)
        endif()
    else()
        message(WARNING "svdquant: Ascend requested but CANN not found — disabling (set ASCEND_HOME_PATH)")
        set(SVDQUANT_ENABLE_ASCEND OFF)
        set(SVDQUANT_ASCENDC_AVAILABLE OFF)
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
