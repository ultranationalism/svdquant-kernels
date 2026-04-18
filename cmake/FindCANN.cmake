# FindCANN.cmake — locate Huawei CANN / AscendC toolkit.
#
# Sets:
#   CANN_FOUND
#   CANN_ROOT
#   CANN_INCLUDE_DIRS
#   CANN_LIBRARIES
#   CANN_ASCENDC_COMPILER  (ccec)
#
# Hints (any of):
#   $ENV{ASCEND_HOME_PATH}
#   $ENV{ASCEND_TOOLKIT_HOME}

set(_cann_hints
    $ENV{ASCEND_HOME_PATH}
    $ENV{ASCEND_TOOLKIT_HOME}
    /usr/local/Ascend/ascend-toolkit/latest
    /usr/local/Ascend/latest
)

find_path(CANN_ROOT
    NAMES include/acl/acl.h
    HINTS ${_cann_hints}
    DOC   "Root of CANN (Ascend) toolkit"
)

if(CANN_ROOT)
    set(CANN_INCLUDE_DIRS
        ${CANN_ROOT}/include
        ${CANN_ROOT}/include/acl
    )

    find_library(CANN_ASCENDCL NAMES ascendcl HINTS ${CANN_ROOT}/lib64 ${CANN_ROOT}/lib)
    find_library(CANN_RUNTIME  NAMES runtime  HINTS ${CANN_ROOT}/lib64 ${CANN_ROOT}/lib)

    find_program(CANN_ASCENDC_COMPILER
        NAMES ccec
        HINTS ${CANN_ROOT}/compiler/ccec_compiler/bin
              ${CANN_ROOT}/tools/ccec_compiler/bin
              ${CANN_ROOT}/bin
    )

    set(CANN_LIBRARIES ${CANN_ASCENDCL} ${CANN_RUNTIME})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CANN
    REQUIRED_VARS CANN_ROOT CANN_INCLUDE_DIRS
)

mark_as_advanced(CANN_ROOT CANN_ASCENDCL CANN_RUNTIME CANN_ASCENDC_COMPILER)
