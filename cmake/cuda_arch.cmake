# SVDQuant CUDA target SM list. Update when a new arch is supported.
#   80  Ampere  (A100)
#   89  Ada     (4090 / L40)
#   90  Hopper  (H100)
#  100  Blackwell (B200) — reserved
if(NOT DEFINED SVDQUANT_CUDA_ARCHS)
    set(SVDQUANT_CUDA_ARCHS "80;89;90" CACHE STRING "CUDA SM archs to build")
endif()

set(CMAKE_CUDA_ARCHITECTURES ${SVDQUANT_CUDA_ARCHS})
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Lets per-file code gate on specific arches, e.g. #ifdef SVDQUANT_HAS_SM90.
foreach(_arch IN LISTS SVDQUANT_CUDA_ARCHS)
    add_compile_definitions(SVDQUANT_HAS_SM${_arch}=1)
endforeach()
