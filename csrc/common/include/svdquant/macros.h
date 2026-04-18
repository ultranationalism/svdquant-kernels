#pragma once

#if defined(__CUDACC__)
  #define SVDQUANT_HOST_DEVICE __host__ __device__
  #define SVDQUANT_DEVICE      __device__
  #define SVDQUANT_HOST        __host__
  #define SVDQUANT_FORCE_INLINE __forceinline__
#else
  #define SVDQUANT_HOST_DEVICE
  #define SVDQUANT_DEVICE
  #define SVDQUANT_HOST
  #define SVDQUANT_FORCE_INLINE inline
#endif

#define SVDQUANT_UNUSED(x) ((void)(x))
