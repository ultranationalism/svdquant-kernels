#!/usr/bin/env bash
# Convenience configure+build. Override any var on the command line, e.g.:
#   BUILD_TYPE=Debug CUDA=OFF ./scripts/build.sh
#
# This is a personal kernel-tuning workbench — the environment assumptions
# baked below (CUDA 13 at /usr/local/cuda-13.0, CANN at
# /usr/local/Ascend/...) are for this host only, not a portable recipe.
set -euo pipefail

# Re-exec with a scrubbed env on first entry. Conda base activates a
# `compilers` metapackage that rewrites 40+ env vars (CC, CXX, CFLAGS,
# LDFLAGS, CONDA_BUILD_SYSROOT, …) to point native builds at conda's
# cu12.8 toolchain and bake /root/miniconda3/lib into DT_RPATH. That
# shadows the host's CUDA 13 and breaks `compute_103` (SM_103 only
# landed in CUDA 12.9+). We scrub with `env -i` and hand-pick the few
# vars we actually want — same pattern the smoke build in
# `tmp/smoke_add/build.sh` uses. See the memory file
# `conda_build_env_pollution.md` for the full post-mortem.
if [[ -z "${SVDQUANT_BUILD_CLEAN_ENV:-}" ]]; then
    exec env -i \
        HOME="${HOME:-/root}" \
        USER="${USER:-root}" \
        TERM="${TERM:-dumb}" \
        LANG="${LANG:-C.UTF-8}" \
        PATH="/usr/local/cuda-13.0/bin:/usr/bin:/bin" \
        ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}" \
        BUILD_DIR="${BUILD_DIR:-}" \
        BUILD_TYPE="${BUILD_TYPE:-}" \
        CUDA="${CUDA:-}" \
        ASCEND="${ASCEND:-}" \
        TESTS="${TESTS:-}" \
        BENCH="${BENCH:-}" \
        SVDQUANT_BUILD_CLEAN_ENV=1 \
        bash "$0" "$@"
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
CUDA="${CUDA:-ON}"
ASCEND="${ASCEND:-ON}"
TESTS="${TESTS:-OFF}"
BENCH="${BENCH:-OFF}"

cmake -S "${ROOT}" -B "${BUILD_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DSVDQUANT_ENABLE_CUDA="${CUDA}" \
    -DSVDQUANT_ENABLE_ASCEND="${ASCEND}" \
    -DSVDQUANT_BUILD_TESTS="${TESTS}" \
    -DSVDQUANT_BUILD_BENCHMARKS="${BENCH}"

cmake --build "${BUILD_DIR}" -j
