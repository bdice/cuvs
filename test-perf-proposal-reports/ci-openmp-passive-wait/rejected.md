# Rejected: `OMP_WAIT_POLICY=passive` for the C++ test runs

**Proposal.** Export `OMP_WAIT_POLICY=passive` in `ci/test_cpp.sh`. With the LLVM OpenMP runtime in the test environment,
this sets `KMP_BLOCKTIME=0`, so idle OpenMP workers sleep right after each parallel region instead of spinning for
200 ms. The idea was that spinning workers in one test process take CPU from the other processes under `ctest -j8`.

**Result: rejected.** It does not shorten the suite; it lengthens it by about 1%. It does cut CPU time by about 75%.

## What was measured

On the final `test-perf-proposals` build (all accepted proposals applied), RTX 6000 Ada, 36-thread host, otherwise idle.

### Whole suite, `ctest -j8` (59 tests; `KVIKIO_COMPAT_MODE=ON` as in CI), runs alternated

| setting | run 1 | run 2 |
|---|---|---|
| default | 857.9 s | 854.2 s |
| `OMP_WAIT_POLICY=passive` | 866.3 s | 865.7 s |

All tests passed in every run. Passive is +1.2% (about 10 s) on the suite.

### Single process, each executable alone, default vs passive alternated, 2 repetitions each (mean)

| executable | wall default | wall passive | change | CPU default | CPU passive | change |
|---|---|---|---|---|---|---|
| `NEIGHBORS_ANN_NN_DESCENT_TEST` | 33.5 s | 44.9 s | +34.1% | 1181 s | 530 s | -55.1% |
| `NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST` | 23.4 s | 23.9 s | +2.3% | 541 s | 105 s | -80.5% |
| `NEIGHBORS_ANN_CAGRA_HALF_UINT32_TEST` | 13.7 s | 14.4 s | +5.1% | 340 s | 77 s | -77.4% |
| `NEIGHBORS_ANN_CAGRA_INT8_UINT32_TEST` | 15.9 s | 16.6 s | +4.1% | 384 s | 86 s | -77.6% |
| `NEIGHBORS_ANN_CAGRA_UINT8_UINT32_TEST` | 16.9 s | 17.5 s | +3.6% | 417 s | 90 s | -78.4% |
| `NEIGHBORS_ANN_HNSW_ACE_FLOAT_UINT32_TEST` | 3.2 s | 4.3 s | +32.9% | 99 s | 35 s | -64.7% |
| `NEIGHBORS_ANN_HNSW_ACE_HALF_UINT32_TEST` | 3.1 s | 4.2 s | +33.8% | 95 s | 35 s | -62.7% |
| `NEIGHBORS_ANN_HNSW_ACE_INT8_UINT32_TEST` | 3.1 s | 4.1 s | +30.0% | 93 s | 35 s | -62.5% |
| `NEIGHBORS_ANN_HNSW_ACE_UINT8_UINT32_TEST` | 3.0 s | 4.1 s | +36.9% | 92 s | 35 s | -62.1% |
| `NEIGHBORS_ANN_CAGRA_BBQ_UINT32_TEST` | 1.9 s | 2.5 s | +28.2% | 49 s | 26 s | -47.3% |
| `NEIGHBORS_ALL_NEIGHBORS_TEST` | 16.8 s | 19.4 s | +15.3% | 565 s | 182 s | -67.8% |
| `NEIGHBORS_HNSW_TEST` | 34.6 s | 29.7 s | -14.2% | 1220 s | 248 s | -79.7% |
| `NEIGHBORS_ANN_IVF_RABITQ_TEST` | 9.5 s | 9.0 s | -5.3% | 202 s | 10 s | -95.2% |
| `NEIGHBORS_ANN_SCANN_TEST` | 5.8 s | 6.3 s | +7.6% | 126 s | 8 s | -93.5% |
| `PREPROCESSING_TEST` | 4.3 s | 4.2 s | -2.8% | 58 s | 6 s | -89.6% |
| `NEIGHBORS_ANN_BRUTE_FORCE_TEST` | 3.7 s | 3.8 s | +0.1% | 3 s | 3 s | +0.0% |
| `NEIGHBORS_ANN_IVF_PQ_TEST` | 136.9 s | 129.8 s | -5.1% | 962 s | 130 s | -86.4% |
| **total** | 329.5 s | 338.6 s | +2.8% | 6428 s | 1642 s | -74.5% |

## Why it doesn't help

* **The suite barely runs in parallel today.** Under `ctest -j8` the per-test times add up to 964 s, against 858 s of
  wall time. That is because most executables are still registered with `PERCENT 100`, so ctest runs them one at
  a time on the single GPU. With no concurrent test processes, a spinning worker does not take CPU from anyone, and
  the per-test times match the single-process ones.
* **Waking sleeping workers costs time** in tests whose host loops alternate quickly with GPU work: NN-descent
  iterations (+34%), the HNSW-ACE partition builds (+30–37%), BBQ (+28%), all-neighbors (+15%). These loops run on one
  hot OpenMP team since the per-iteration thread was removed (`../nn-descent-host-loop/`), and with
  `KMP_BLOCKTIME=0` that team goes to sleep between the regions of every iteration.
* Executables that spin for nothing during long GPU phases do get faster: HNSW −14%, IVF-PQ −5%, RaBitQ −5%. But
  that does not make up for the losses.

## When to revisit

* Once the per-executable `PERCENT` values (the test-split PR) let ctest run several test processes on the GPU at
  once, CPU contention between processes becomes real. The −75% CPU time would then matter, especially on CI runners
  with fewer cores than this 36-thread host.
* A middle ground to measure then: a short `KMP_BLOCKTIME` such as 1–5 ms. That keeps a team hot within an
  NN-descent iteration but stops the 200 ms spin after a test's host phase. It is libomp-specific; with GNU libgomp
  the equivalent is `GOMP_SPINCOUNT`.
