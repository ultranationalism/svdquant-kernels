# Per-pod numerical correctness tests. Intentionally empty at scaffold time.
#
# When a kernel lands, drop a `tests/kernels/<op>/` directory with a
# `CMakeLists.txt` wiring a small executable against the pod's OBJECT lib
# (`svdquant::<op>`) and the baseline under `baseline/kernels/<op>/` as
# numerical ground truth. Register it with `add_test(...)`.

# add_subdirectory(kernels/w4a4_gemm)
# add_subdirectory(kernels/lowrank_branch)
# ...
