#!/usr/bin/env python3
# Drop-in for CANN's ASCEND_PYTHON_EXECUTABLE.
#
# Works around a CANN 8.5 extract_host_stub.py bug: when an ascendc
# device source lives outside the sub-build's CMAKE_SOURCE_DIR (which
# is hardcoded to <toolkit>/tikcpp/.../device_preprocess_project/),
# CMake encodes the .o path with a `/./` disambiguator. But
# compile_commands.json records the same .o without `/./`. The Python
# script (extract_host_stub.py) builds its lookup dict from
# compile_commands and queries it with the raw .o path it receives on
# the command line — the two forms don't match and it raises KeyError.
#
# Fix: normpath() any `/./` or `/../` segments in argv before exec'ing
# real python3. Harmless for paths that don't need it.
import os
import sys

REAL_PYTHON = sys.executable if os.path.basename(sys.executable) != os.path.basename(__file__) \
              else "/usr/bin/python3"

new_argv = []
for arg in sys.argv[1:]:
    if "/./" in arg or "/../" in arg:
        new_argv.append(os.path.normpath(arg))
    else:
        new_argv.append(arg)

os.execv(REAL_PYTHON, [REAL_PYTHON] + new_argv)
