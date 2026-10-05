# NEIGHBORS_ANN_CAGRA_FILTER_UDF_TEST: where the time goes and how to make it faster

Source: 8 `nsys profile -t cuda,nvtx,osrt` runs, one TEST_P per process (3 search algos each, 24 cases), RTX 6000 Ada,
branch `staging-test-optimizations-local`. Test: `cpp/tests/neighbors/ann_cagra/test_filter_udf.cu`. All times are gtest
times under nsys. "JIT" is the host gap before each rtcx `cudaLibraryLoadData`, i.e. NVRTC + nvJitLink + cache I/O.

## Summary

| TEST_P (all `CagraUdfFilters/CagraUdfFilterTest` except last) | gtest s | JIT s | index build s | other s |
|---|---|---|---|---|
| AcceptAllMatchesNoFilter | 2.68 | 2.10 | 0.30 | 0.28 |
| RejectAllReturnsNoValidNeighbors | 2.51 | 1.97 | 0.36 | 0.18 |
| HighFilteringRateReturnsOnlyValidNeighbors | 2.50 | 2.06 | 0.28 | 0.16 |
| RepeatedUdfSearchWithSameSourceMatches (UDF already in JIT disk cache) | 0.75 | 0.20 | 0.35 | 0.20 |
| InvalidSourceThrows (NVRTC fails, nothing linked) | 0.56 | 0.02 | 0.30 | 0.24 |
| ThresholdMatchesEquivalentBitset | 2.60 | 2.08 | 0.30 | 0.22 |
| TenantContextHonorsQuerySpecificMetadata | 2.63 | 2.14 | 0.31 | 0.18 |
| `CagraUdfFilterHalf/...ThresholdReturnsOnlyValidNeighbors` | 2.64 | 2.19 | 0.27 | 0.18 |
| **Total (24 cases)** | **16.9** | **12.8 (76%)** | **2.5 (15%)** | **1.6 (9%)** |

GPU is idle: 0.13 s of kernels in total (<1% busy). Each search takes <1 ms.

## Where the time goes

* **nvJitLink LTO of the search kernel, once per new UDF source: ~12.2 s (72%).** The fragment key contains the source
  text (`src/neighbors/detail/cagra/jit_lto_kernels/sample_filter_udf.cuh:78-86`), so every distinct UDF re-links the
  whole kernel. Per (source, dtype) on a cache miss: SINGLE_CTA 1.37-1.45 s, MULTI_CTA 0.44-0.52 s, MULTI_KERNEL
  0.10-0.21 s (2 kernels), ~2.0 s in total (2.2 s for half). The link is single-threaded CPU work: no API/OS calls on
  the main thread (1.28 s in AcceptAll/0), with the OpenMP pools idle in `pthread_cond_wait`.
* **NVRTC compile of the user source is small:** ~70-85 ms per source per process (the gap before nvJitLink's
  `cuInit`), ~0.5 s in total (3%). The failing compile in InvalidSourceThrows takes 20-40 ms.
* **Links are cached in `~/.nv/ComputeCache`, but the cache is full.** Each miss group wrote 4 entries
  (770/381/232/38 KB) whose mtimes match the link ends within ±10 ms. RepeatedUdf reused AcceptAll's source in a later
  process and hit: 0.20 s for 3 algos vs. 1.99 s. The cache holds 1,073,442,295 B in 2239 entries, i.e. it is **at the
  1 GiB default `CUDA_CACHE_MAXSIZE`** and evicts LRU, so UDF kernels from earlier runs were gone. Non-UDF filter links
  (none, bitset, roaring) hit: 20-40 ms each.
* **Index build: 2.5 s (15%), 24 rebuilds of the same index.** It is 768x16 NN-descent with seed 1234 and the same params
  in every case (`test_filter_udf.cu:113-134`, half :182-213). Steady state is ~60 ms per build (33-98 ms). The first
  case in each process takes 160-230 ms because of lazy module loading. General NN-descent costs (a `std::thread` per
  iteration) are covered in `NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST.md`.
* **Per-process start-up (an artifact of one process per group):** ~0.33 s before main, plus 0.11-0.17 s of CUDA
  context and pool init inside the first case. That adds ~1.1 s of gtest time over 8 processes.
* **One ctest process (projection):** 0.5 s start-up + 6 distinct (source, dtype) link sets x ~2.05 s + builds ~1.6 s +
  ~0.5 s = **~15 s** (RepeatedUdf reuses AcceptAll's launchers via the in-process cache, rtcx
  `algorithm_planner.cpp:42-55`). That matches the 15.6 s logged under `ctest -j8` with MPS; a warm JIT cache gives ~4 s.

## Ideas (ranked by estimated savings vs. the ~15 s single-process run; coverage-reducing alternative last)

1. **Keep the CUDA JIT cache warm. No code change.** Set `CUDA_CACHE_MAXSIZE=4294967296` (the cache is full at 1 GiB
   now). In CI, also persist `CUDA_CACHE_PATH` across jobs (keyed on the libcuvs build). A cached UDF costs 0.2 s
   instead of 2.0 s: **~10.8 s (72%)** on its own, ~5.4 s after idea 2. It also helps every JIT-LTO test executable.
   Test asserts are unchanged, but the cold nvJitLink path then only runs on a miss. Effort low (env var) to medium
   (CI cache); risk low.
2. **Use one parameterized UDF source instead of four. Keeps coverage.** Replace `accept_all`/`reject_all`/
   `high_filtering_rate`/`threshold` (`test_filter_udf.cu:52-84`) with
   `return source_id >= *static_cast<const uint32_t*>(filter_data);` and pass a device scalar of 0 / 768 / 704 / 192
   at :264, :286, :299-300, :312, :342-343. Distinct float sources drop from 5 to 2: **~6 s (40%)**. Each test still
   checks its semantics on all 3 algos; only source-text variety and the `filter_data == nullptr` path are lost.
   Keeping `accept_all` (nullptr) gives ~4 s (27%); also expressing the threshold as a tenant table
   (`row_tenants[i] = i >= T`) leaves one float source: ~8 s (53%). Caveat: MULTI_CTA raises itopk with
   `filtering_rate` (`search_plan.cuh:224-235`); if that picks other fragments, MULTI_CTA still links once per rate
   (saving ~4.6 s). Effort low, risk low. Result: ~9 s with a cold cache, ~3.2 s with idea 1.
3. **Library side, out of test scope:** a 1-line UDF forces a 1.3 s re-LTO of a 770 KB SINGLE_CTA cubin. Options: try
   nvJitLink `-split-compile` through `linktime_extra_options` (rtcx `algorithm_planner.cpp:71-77`; the gain is
   uncertain for a single kernel), or link the UDF as a separately compiled, non-inlined function. Up to ~10 s here (upper
   bound, unmeasured), and the same latency for users. Effort high, risk medium (search speed).
4. **Build each fixture's index once per suite. Keeps coverage.** Use `SetUpTestSuite` with a static index for the
   float and half fixtures (:113-134, :182-213) and free it in `TearDownTestSuite`; the index is const during search.
   24 builds become 2: **~1.3 s (9%)**. Effort low, risk low.
5. **Reduces coverage, not recommended:** as an alternative to idea 2, run the Reject/HighFiltering/Threshold UDF tests
   on MULTI_KERNEL only (split INSTANTIATE at :456-466). This saves ~1.85 s per source (~5.5 s) and is redundant after
   idea 2. Also keep the executable in one process: per-group sharding adds ~0.6 s of start-up and lazy loading per
   process, and it loses the in-process launcher cache.

## Uncertainties

* No CPU sampling, so the NVRTC/nvJitLink split comes from gap structure (`cuInit` and cache `fopen` markers). The
  claim that nvJitLink writes to ComputeCache comes from file mtimes that match the link ends.
* The non-UDF filter links hit the cache here. On a fully cold cache (fresh CI container), the none, bitset and roaring
  variants could add up to ~3 x 2 s more, which makes idea 2 even larger.
* Logged `ctest -j8` times for this executable vary from 15.6 s (MPS) to 27.9, 48.9 (the quoted 49 s) and 86.6 s. Cache
  hits vs. misses plus GPU time-slicing of ~1-4k tiny launches and ~400 syncs per group are likely causes, but runs
  under contention were not profiled.
* The savings assume link cost does not depend on the UDF body: SINGLE_CTA was 1.37-1.45 s for all 6 sources.
