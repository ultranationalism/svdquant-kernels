#!/usr/bin/env bash
# Source before configuring CMake on an Ascend machine:
#   source scripts/env_ascend.sh
# Defaults to the stock CANN install path; override ASCEND_HOME_PATH to relocate.

export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
if [[ -f "${ASCEND_HOME_PATH}/bin/setenv.bash" ]]; then
    # shellcheck disable=SC1091
    source "${ASCEND_HOME_PATH}/bin/setenv.bash"
elif [[ -f "${ASCEND_HOME_PATH}/set_env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${ASCEND_HOME_PATH}/set_env.sh"
else
    echo "env_ascend: no setenv.bash / set_env.sh under ${ASCEND_HOME_PATH}" >&2
    return 1 2>/dev/null || exit 1
fi

echo "env_ascend: ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
