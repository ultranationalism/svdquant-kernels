# Per-pod numerical correctness tests. Intentionally empty at scaffold time.
#
# When a native kernel lands, drop a `tests/kernels/<op>/` directory with
# a `CMakeLists.txt` wiring a small executable against the pod's OBJECT
# lib (`svdquant::<op>`) and the baseline under `baseline/kernels/<op>/`
# as numerical ground truth. Register it with `add_test(...)`.
#
# Triton pods (`triton_kernels/<op>/`) aren't tested from CMake — test
# them from pytest on the host where the Triton runtime is installed.

# add_subdirectory(kernels/gemm_w4a4)
