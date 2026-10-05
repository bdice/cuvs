### Most C++ tests reserve the whole GPU, so `ctest -j8` runs them one at a time

**Problem**

ctest only runs tests concurrently if their `RESOURCE_GROUPS` fit on the GPU. In `cpp/tests/CMakeLists.txt`, 34 of the
38 C++ test executables are registered with `PERCENT 100` (exceptions: ScaNN at 1, Vamana at 10, and the MG tests),
so each of them takes the whole GPU. Measured on an RTX 6000 Ada with the full suite at `ctest -j8`, the per-test times
add up to about 962 s against 856 s of wall time: almost no overlap.

Most of these executables need little memory. `cpp/scripts/gtest_memory_usage.sh` reports a peak RMM usage under
1 GiB for 29 of the 36 single-GPU executables; the largest are `DISTANCE_TEST` (12.3 GiB), `CLUSTER_TEST` (7.5 GiB),
`NEIGHBORS_TEST` (3.2 GiB) and `NEIGHBORS_ANN_IVF_FLAT_TEST` (1.7 GiB).

**Proposal**

Set each executable's `PERCENT` to its measured peak as a share of a 16 GB GPU (the smallest supported), rounded up to
a multiple of 5 (minimum 5), and keep `RUN_SERIAL` tests at 100. Re-measure with `gtest_memory_usage.sh` when tests
change.

**Also noticed**

* `gtest_memory_usage.sh` globs `gtests/*_TEST`, so it skips `NEIGHBORS_ANN_CAGRA_TEST_BUGS`.
* The reported peak is RMM-only, so it doesn't include the CUDA context.
