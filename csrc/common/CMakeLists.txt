add_library(svdquant_common INTERFACE)
target_include_directories(svdquant_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(svdquant_common INTERFACE cxx_std_17)
add_library(svdquant::common ALIAS svdquant_common)
