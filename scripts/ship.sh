#!/usr/bin/env bash
# Stage an artifact (file or dir) under tmp/ and upload to the OpenI dataset
# that the remote serverless NPU instance auto-extracts at boot.
#
#   scripts/ship.sh <path>                 # tar+upload (dir) or upload (file)
#   scripts/ship.sh <path> <name>          # override uploaded filename
#   REPO=other/name scripts/ship.sh <path> # override dataset
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="${REPO:-ultranationalism/svdquant-ops}"
STAGE="${ROOT}/tmp"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <path> [name]" >&2
    exit 1
fi
SRC="$1"
NAME="${2:-}"

if [[ ! -e "$SRC" ]]; then
    echo "ship: no such path: $SRC" >&2
    exit 1
fi

mkdir -p "$STAGE"

if [[ -d "$SRC" ]]; then
    STAMP="$(date +%Y%m%d-%H%M%S)"
    TARNAME="${NAME:-$(basename "$SRC")-${STAMP}.tar.gz}"
    TAR="${STAGE}/${TARNAME}"
    echo ">> packing $SRC → $TAR"
    tar -czf "$TAR" -C "$(dirname "$SRC")" "$(basename "$SRC")"
    UPLOAD="$TAR"
else
    UPLOAD="$SRC"
fi

echo ">> uploading $UPLOAD → dataset $REPO"
if [[ -n "$NAME" && -f "$UPLOAD" ]]; then
    openi dataset upload "$REPO" "$UPLOAD" -n "$NAME"
else
    openi dataset upload "$REPO" "$UPLOAD"
fi
