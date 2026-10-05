### Size each C++ test's GPU share from its measured peak memory

Almost every C++ test executable was registered with `PERCENT 100`, so `ctest -j8` ran them one at a time on the GPU.
This PR sets each executable's `PERCENT` in `cpp/tests/CMakeLists.txt` from its peak RMM memory. The peaks were
measured with `cpp/scripts/gtest_memory_usage.sh`.

* `PERCENT` = peak / 16 GiB (the smallest GPU we support), rounded up to a multiple of 5, minimum 5. For example,
  8 GiB → 50 and 1.7 GiB → 15.
* Tests with `RUN_SERIAL` (the MG tests) keep `PERCENT 100`; ctest runs them alone anyway.
* Result: 29 executables get 5, three get 10, and the rest get 15 (IVF-Flat), 20 (`NEIGHBORS_TEST`), 50
  (`CLUSTER_TEST`) and 80 (`DISTANCE_TEST`). The full table and the raw script output are in the notes.

The script reports the RMM peak only, without the CUDA context. With `-j8`, at most 8 processes share the GPU.

## Testing

The full suite passes with the new values in every run below (59 tests; the install-header check is excluded because
it needs installed headers).

## Measurements

Whole C++ suite, `ctest -j8`, RTX 6000 Ada (48 GB), 36-thread host, warm JIT cache, `KVIKIO_COMPAT_MODE=ON` as in CI.
Same build for both; only the generated `tests/CTestTestfile.cmake` differs. Runs alternated old, new, new, old:

| PERCENT values | run 1 | run 2 | mean | sum of per-test times |
|---|---|---|---|---|
| before (most at 100) | 841.8 s | 826.0 s | 833.9 s | 953 / 936 s |
| after (from peak memory) | 708.5 s | 710.3 s | **709.4 s (−14.9%)** | 4,736 / 4,751 s |

The tests now overlap: their summed run time is 6.7× the wall time instead of 1.1×. Each test runs slower when it
shares the GPU and the host with up to 7 others. IVF-PQ goes from 138 s alone to 672 s, IVF-Flat from 77 s to 607 s,
and Vamana from 252 s to 439 s. So the wall time is now set by those long tests under contention, plus the two MG
tests that run serially at the end (≈ 38 s).

Closes #`<issue>`
