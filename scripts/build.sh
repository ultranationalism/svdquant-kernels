#!/usr/bin/env bash
# Convenience configure+build. Override any var on the command line, e.g.:
#   BUILD_TYPE=Debug CUDA=OFF ./scripts/build.sh
set -euo pipefail

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
