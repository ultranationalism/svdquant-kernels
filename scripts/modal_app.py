"""Run svdquant CUDA kernels on a Modal Blackwell (B200) instance.

Prereq:
    ./scripts/build.sh           # build SM_100 artifacts into ./build/

Usage:
    modal run scripts/modal_app.py              # smoke: nvidia-smi + torch + ls build/
    modal run scripts/modal_app.py::tests       # run every executable under build/

`build/` is mounted read-only at `/root/build` inside the container via
`add_local_dir(copy=False)` — files land at container start, not in an
image layer, so each run picks up the latest local build without
rebuilding the image.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT / "build"
BUILD_DIR.mkdir(exist_ok=True)

app = modal.App("svdquant-kernels")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime")
    .add_local_dir(str(BUILD_DIR), remote_path="/root/build", copy=False)
)


@app.function(gpu="B200", image=image)
def smoke() -> None:
    subprocess.run(["nvidia-smi"], check=True)
    import torch

    print(f"torch {torch.__version__}, cuda {torch.version.cuda}")
    print(
        f"device: {torch.cuda.get_device_name(0)}, "
        f"sm{''.join(map(str, torch.cuda.get_device_capability(0)))}"
    )
    root = Path("/root/build")
    files = sorted(p for p in root.rglob("*") if p.is_file())
    print(f"build/ contains {len(files)} files:")
    for p in files:
        print(f"  {p.relative_to(root)}  ({p.stat().st_size} B)")


@app.function(gpu="B200", image=image)
def tests() -> None:
    root = Path("/root/build")
    bins = [
        p for p in root.rglob("*")
        if p.is_file() and os.access(p, os.X_OK) and p.suffix == ""
    ]
    if not bins:
        print("no executables under build/ — rebuild with TESTS=ON?")
        return
    for b in bins:
        print(f"==> {b.relative_to(root)}")
        subprocess.run([str(b)], check=False)


@app.local_entrypoint()
def main() -> None:
    smoke.remote()
