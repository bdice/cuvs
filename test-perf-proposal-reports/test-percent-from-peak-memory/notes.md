# test-percent-from-peak-memory: size each C++ test's GPU share from its peak memory

Commit 8228931d on `test-perf-proposals` (`cpp/tests/CMakeLists.txt` only).

## Why

Under `ctest -j8`, ctest only runs tests concurrently if their `RESOURCE_GROUPS` fit on the GPU. Almost every C++ test
executable was registered with `PERCENT 100`, so the suite ran almost serially: the per-test times added up to about
962 s against 856 s of wall time (see `../SUMMARY.md`).

## Method

1. Built `test-perf-proposals` at 8d72304f (all other proposals applied).
2. Ran `cpp/scripts/gtest_memory_usage.sh` from a directory whose `gtests/` held only the 38 executables registered in
   the current `tests/CTestTestfile.cmake`. The build directory also holds stale executables from earlier experiments.
   `GTEST_CUVS_RMM_MODE` was unset (async), and `KVIKIO_COMPAT_MODE=ON` was set as in CI. RTX 6000 Ada.
3. The script globs `gtests/*_TEST`, which misses `NEIGHBORS_ANN_CAGRA_TEST_BUGS`. That executable was measured the
   same way: `GTEST_CUVS_MEMORY_PEAK=1 GTEST_BRIEF=1 gtests/NEIGHBORS_ANN_CAGRA_TEST_BUGS | grep Peak`.
4. `PERCENT = 5 * ceil(peak / 16 GiB * 100 / 5)`, with a minimum of 5. So 8 GiB → 50 and 1.7 GiB → 15.
5. Tests with `RUN_SERIAL` (the MG tests) keep `PERCENT 100`. ctest runs them alone anyway.

Raw script output: `gtest_memory_usage.csv`.

## Result

| executable | peak RMM memory | PERCENT before | PERCENT after | note |
|---|---|---|---|---|
| `NEIGHBORS_MG_TEST` | 38,414 MiB | 100 | 100 | RUN_SERIAL, unchanged |
| `CLUSTER_KMEANS_MG_TEST` | 24,009 MiB | 100 | 100 | RUN_SERIAL, unchanged |
| `DISTANCE_TEST` | 12,601 MiB | 100 | 80 |  |
| `CLUSTER_TEST` | 7,648 MiB | 100 | 50 |  |
| `NEIGHBORS_TEST` | 3,228 MiB | 100 | 20 |  |
| `NEIGHBORS_ANN_IVF_FLAT_TEST` | 1,730 MiB | 100 | 15 |  |
| `NEIGHBORS_EPSILON_NEIGHBORHOOD_TEST` | 1,145 MiB | 100 | 10 |  |
| `NEIGHBORS_ANN_BRUTE_FORCE_TEST` | 1,010 MiB | 100 | 10 |  |
| `NEIGHBORS_ANN_CAGRA_TEST_BUGS` | 843 MiB | 100 | 10 |  |
| `PREPROCESSING_TEST` | 788 MiB | 100 | 5 |  |
| `NEIGHBORS_ANN_IVF_SQ_TEST` | 768 MiB | 100 | 5 |  |
| `NEIGHBORS_ANN_IVF_PQ_TEST` | 681 MiB | 100 | 5 |  |
| `NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST` | 396 MiB | 100 | 5 |  |
| `NEIGHBORS_ANN_CAGRA_HALF_UINT32_TEST` | 395 MiB | 100 | 5 |  |
| `NEIGHBORS_ANN_CAGRA_INT8_UINT32_TEST` | 395 MiB | 100 | 5 |  |
| `NEIGHBORS_ANN_CAGRA_UINT8_UINT32_TEST` | 395 MiB | 100 | 5 |  |
| `NEIGHBORS_DYNAMIC_BATCHING_TEST` | 301 MiB | 100 | 5 |  |
| `NEIGHBORS_ALL_NEIGHBORS_TEST` | 203 MiB | 100 | 5 |  |
| `NEIGHBORS_ANN_CAGRA_MERGE_TEST` | 196 MiB | 100 | 5 |  |
| `NEIGHBORS_ANN_SCANN_TEST` | 166 MiB | 1 | 5 |  |
| `NEIGHBORS_ANN_IVF_RABITQ_TEST` | 110 MiB | 100 | 5 |  |
| `NEIGHBORS_ANN_NN_DESCENT_TEST` | 81 MiB | 100 | 5 |  |
| `NEIGHBORS_TIERED_INDEX_TEST` | 70 MiB | 100 | 5 |  |
| `STATS_TEST` | 64 MiB | 100 | 5 |  |
| `NEIGHBORS_ANN_VAMANA_TEST` | 19 MiB | 10 | 5 |  |
| `NEIGHBORS_ANN_HNSW_ACE_FLOAT_UINT32_TEST` | 19 MiB | 100 | 5 |  |
| `NEIGHBORS_HNSW_TEST` | 18 MiB | 100 | 5 |  |
| `NEIGHBORS_ANN_HNSW_ACE_HALF_UINT32_TEST` | 18 MiB | 100 | 5 |  |
| `NEIGHBORS_ANN_HNSW_ACE_INT8_UINT32_TEST` | 17 MiB | 100 | 5 |  |
| `NEIGHBORS_ANN_HNSW_ACE_UINT8_UINT32_TEST` | 17 MiB | 100 | 5 |  |
| `NEIGHBORS_ANN_CAGRA_BBQ_UINT32_TEST` | 10 MiB | 100 | 5 |  |
| `NEIGHBORS_BALL_COVER_TEST` | 9 MiB | 100 | 5 |  |
| `UTIL_TEST` | 3 MiB | 100 | 5 |  |
| `NEIGHBORS_ANN_CAGRA_FILTER_UDF_TEST` | 1 MiB | 100 | 5 |  |
| `CORE_ROARING_ALLOWLIST_TEST` | 0 MiB | 100 | 5 |  |
| `NEIGHBORS_ANN_CAGRA_HELPERS_TEST` | 0 MiB | 100 | 5 |  |
| `NEIGHBORS_ANN_IVF_FLAT_UDF_TEST` | 0 MiB | 100 | 5 |  |
| `CUTILE_SMOKE_TEST` | 0 MiB | 100 | 5 |  |
| `CLUSTER_KMEANS_MNMG_TEST` | — | 100 | 100 | not measured |

`CLUSTER_KMEANS_MNMG_TEST` is not built in this configuration.

## Caveats

* **RMM peak only.** `GTEST_CUVS_MEMORY_PEAK` reports the peak of RMM allocations. It does not count the CUDA context
  (a few hundred MB per process), cuBLAS/cuSOLVER workspaces allocated outside RMM, or loaded JIT kernels. So the
  values follow the requested rule (peak / 16 GB) but leave little headroom for tests whose RMM peak is just under a
  5% step. Example: `PREPROCESSING_TEST` uses 788 MiB and gets 5% = 819 MiB. With `-j8` at most 8 processes share the
  GPU, so on a 16 GB GPU the contexts add a few GB on top of the reserved shares.
* **The MG tests don't fit a 16 GB GPU as written.** `CLUSTER_KMEANS_MG_TEST` peaks at 24 GB and `NEIGHBORS_MG_TEST`
  at 38 GB of RMM memory on this 48 GB GPU. They are `RUN_SERIAL` and keep `PERCENT 100`, but on a 16 GB GPU they
  would need smaller inputs or would have to adapt to the available memory.
* **The C API tests** (`c/tests`) are not covered by the script and keep the default `PERCENT 30`.
* **Relation to #2723** (on hold): that PR split long executables and set PERCENT the same way, with a 1.2× headroom
  factor. This commit sets PERCENT for the current executables only, with no headroom factor, as requested.
