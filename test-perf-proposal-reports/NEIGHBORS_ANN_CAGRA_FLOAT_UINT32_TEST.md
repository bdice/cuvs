# NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST: where the time goes and how to make it faster

Source: 35 `nsys profile -t cuda,nvtx,osrt` runs, one fixture group (or gtest shard) per process, on an RTX 6000 Ada,
branch `staging-test-optimizations-local` (includes #2685 and #2726). All times are gtest times measured under nsys.
The sum is 366 s; for comparison, the whole executable took 313 s under `ctest -j8` with other tests sharing the GPU.
Tests: `cpp/tests/neighbors/ann_cagra/test_float_uint32_t.cu`. Fixtures: `cpp/tests/neighbors/ann_cagra.cuh` (abbreviated `.cuh` below).

## Summary

| TEST_P (instantiation) | cases (skipped) | gtest s | GPU busy | kernel launches | dominant cost |
|---|---|---|---|---|---|
| AnnCagraIndexMergeTest / `AnnCagraIndexMerge_U32` | 459 (197) | 77.7 | 17% | ~2.6 M | IVF-PQ graph build 45.5 s |
| AnnCagraIndexMergeTest / `AnnCagraIndexMerge_I64` | 459 (197) | 75.9 | 17% | ~2.6 M | same builds as `_U32` (44.5 s) |
| AnnCagraTest / `AnnCagra_I64` | 459 (161) | 53.7 | 12% | 1.13 M | IVF-PQ 21.5 s, serialize 12.1 s |
| AnnCagraTest / `AnnCagra_U32` | 459 (161) | 51.7 | 12% | 1.13 M | same builds as `_I64` |
| AnnCagraIndexFilteredMergeTest | 459 (324) | 49.2 | 18% | 1.69 M | IVF-PQ 25.8 s (3 builds per case) |
| AnnCagraFilterTest | 60 (0) | 22.7 | 36% | 1.91 M | MULTI_KERNEL search, one query at a time: 14 s |
| AnnCagraMultiPartitionTest Search + FilteredSearch | 24 + 24 | 13.7 + 9.8 | 12% | 0.05 M | NN-descent 15 s (same partitions rebuilt) |
| AnnCagraAddNodesTest | 102 (24) | 8.6 | 73% | 0.10 M | iterative build, one query at a time: 5.2 s |
| 15 non-parameterized tests (CagraQ*, MultiPartition rejects) | 15 | 2.2 | 3% | 0.01 M | start-up |
| **Total** | **2520 (1064, 0.5 s)** | **366** | **18%** (66 of 369 s) | **11.2 M** | host / launch bound |

## Where the time goes

Method: each case was cut out of the timeline using the two `GenerateRoundingErrorFreeDataset_kernel` launches in SetUp. Its
main-thread time was then split into phases by the family of the kernels launched. JIT link time (the host gap before
`cu/cudaLibraryLoadData`) and kvikio serialize/deserialize spans were measured separately.

| bucket | s | % | detail |
|---|---|---|---|
| IVF-PQ graph build: k-means + PQ codebook training | 158.4 | 43 | ~100 ms per build for 1000x16 data, 760 ms at dim 1024. Each build launches ~3k tiny kernels with GPU <10% busy. It runs `pq_dim` (16 to 512) separate balanced k-means, and every k-means iteration does a D2H copy plus a stream sync. |
| NN-descent graph build | 46.9 | 13 | 818 builds, ~54 ms each. Each of ~20 iterations creates and joins a `std::thread`. MultiPartition (10k rows) accounts for 15 s of this. |
| CAGRA search (the test's own searches) | 45.5 | 12 | **40.7 s comes from cases with `dim % 4 != 0`** (dims 1, 3, 7, 17, 102, 137). These take a fallback that runs one plan per query: 100 launches, or for MULTI_KERNEL 100 x ~260 iterations x (4 kernels + D2H + sync). |
| JIT link (nvJitLink) | 32.5 | 9 | ~50 loads per shard at ~20 ms each. Most of this comes from the per-group processes; one ctest process would pay a large share only once. |
| CAGRA graph optimize | 26.6 | 7 | 14.6 s is the reverse-graph loop: per graph column, an OpenMP gather, H2D copy, kernel and sync (4.6 ms x 3144 runs). |
| serialize + deserialize (AnnCagraTest) | 24.0 | 7 | 20.7 ms per case (median). 15.5 ms of that is a host-only gap right after the kvikio file open; 8 x ~1 s is `CUFileInit`, once per process. |
| searches inside the iterative (CAGRA-search) build | 17.6 | 5 | 17.1 s from unaligned dims, caused by the same one-query-at-a-time fallback. |
| test reference + verification + data generation | ~4.5 | 1.2 | `naive_knn`, `eval_distances`, `eval_neighbours`, InitDataset. **Not worth optimizing.** |
| other (IVF-PQ search during build, merge scaffold, glue) | ~7 | 2 | |

Redundancy found in the source:
* **The U32 and I64 TEST_Ps build identical indices.** `AnnCagra_U32` and `_I64` (test_float_uint32_t.cu:16-17), and
  `AnnCagraIndexMerge_U32` and `_I64` (:26-27), differ only in the output index type of `search`. Build, merge and
  serialize are repeated in full: about 105 s.
* **Axes a fixture ignores still double its builds.** AnnCagraTest ignores `merge_strategy` and `host_dataset`. The
  latter only adds a D2H copy, because ACE is never selected (.cuh:467-476). It also ignores `itopk_size` (never set at
  .cuh:451-455) and `search_width` (unused everywhere). This yields 108 identical cases per TEST_P (29 s); the axes are
  at .cuh:1824, 1852, 1881, 1930 and 1949. AnnCagraIndexFilteredMergeTest never sets `graph_degree` (.cuh:1263-1289),
  which gives 20 duplicate cases (6.6 s). It looks like a bug that the degree sweep does nothing there.
* **The same index is rebuilt for every search variant.** Index builds depend only on (n_rows, dim, metric, degree,
  build_algo, refine), but each search-algo / max_queries / team_size / merge-strategy case rebuilds. That is 596
  AnnCagraTest cases for 107 distinct builds, and 524 merge cases for 101. AnnCagraMultiPartitionTest does 48 builds for
  13 distinct partition sets: SINGLE vs MULTI_CTA and Search vs FilteredSearch all share builds.
* **Merge cost multiplies.** Physical merges for InnerProduct/Cosine, for filtered merges, and for n_rows ≤ graph_degree
  fall back to `merge_rebuild` (`cpp/src/neighbors/detail/cagra/cagra_merge.cuh:243-252, 351`). That means three full
  graph builds per case, in addition to the 2x from the U32/I64 split.

## Ideas (ranked by estimated savings; % of the 366 s total; savings overlap, so a combined figure is given)

1. **Build each index once and reuse it for every search variant. Keeps coverage.** Test-only change in `.cuh`. Steps
   in order of effort:
   a. *Fold `AnnCagraIndexMerge_I64` into `_U32`.* Build and merge once, then search and verify into both uint32 and
      int64 outputs (test_float_uint32_t.cu:26-27, .cuh:1424-1631). **61 s (17%)**. Low effort, low risk.
   b. *Fold `AnnCagra_I64` into `AnnCagra_U32`* (test_float_uint32_t.cu:16-17, .cuh:367-549). Also serializes once.
      **43 s (12%)**, or 29 s if 1c is done first. Low effort, low risk.
   c. *Remove cases a fixture cannot distinguish.* Give AnnCagraTest and FilteredMerge their own input lists, or filter
      `inputs` at INSTANTIATE (test_float_uint32_t.cu:36, 47) by an "effective key". Also fix FilteredMerge to honour
      `graph_degree`, or drop that axis. **39 s (11%)** on its own, ~20 s after 1a+1b. Low effort, no risk.
   d. *Reuse the two half-indices between the PHYSICAL and LOGICAL twins* in AnnCagraIndexMergeTest: run both strategies
      in one body, or use the cache in 1e. **17 s (5%)**.
   e. *General alternative:* a process-wide index cache keyed by (dataset shape/metric/seed, slice, graph_degree,
      build_algo, refine), used in AnnCagraTest / IndexMerge / FilteredMerge. Every search, serialize, merge and
      verification still runs against it. This also collapses the corner-case block (.cuh:1806-1826): its IVF_PQ cases
      cost 50 s, built 3x for the search algos and 2x for merge strategy.
   1a-1d together: **~157 s (43%)**. The full cache (1e): **~185 s (51%)**. Effort: medium.
   Risk: shared state across cases, and a failing build would fail every case that uses it. Mitigate by keying strictly
   and clearing the cache in `TearDownTestSuite`. This also speeds up the half / int8 / uint8 executables that share `.cuh`.
2. **Library: train IVF-PQ codebooks in a batch instead of `pq_dim` serial balanced k-means runs.** The serial loop is
   at `cpp/src/neighbors/ivf_pq/ivf_pq_build.cuh:349-416`. Each run syncs every iteration in `adjust_centers`
   (`cpp/src/cluster/detail/kmeans_balanced.cuh:811-812`, loop at :967). The cost pool is 158 s. If half of it is
   launch/sync overhead, that is ~80 s (22%) on its own and ~35 s after idea 1. Keeps coverage. Effort high; risk medium,
   because codebooks and recall may shift. Note the graph-build defaults: `pq_dim` 16, `kmeans_n_iters` 10
   (`cpp/include/cuvs/neighbors/ivf_pq.hpp:3405-3445`).
3. **Library: stop searching one query at a time when `dim % 4 != 0`.** Pass the query row stride into the kernels
   (`queries_ptr += dim * query_id` at `cpp/src/neighbors/detail/cagra/jit_lto_kernels/setup_workspace_impl.cuh:53,143`,
   plus multi-kernel / random_pickup). Then drop the per-query loop at `cagra_search.cuh:114-160`. Saves **~54 s (15%)**:
   40.7 s of search plus 17.1 s of iterative builds, minus the batched cost. After idea 1 it is ~45 s. It also removes
   the 14 s MULTI_KERNEL hotspot in FilterTest. Keeps coverage. Effort medium (signatures of JIT kernels change); risk
   medium. A cheaper partial fix for MULTI_KERNEL alone: check `terminate_flag` every N iterations instead of doing a
   D2H plus sync on every iteration (`search_multi_kernel.cuh:587-592`).
4. **AnnCagraMultiPartitionTest: build each partition set once.** Today 48 builds cover 13 distinct partition sets.
   Cache by (n_rows, dim, num_partitions, split, metric) in `buildPartitions` (.cuh:2153-2173), or fold
   `FilteredSearch` into `Search` (test_float_uint32_t.cu:52-53; that alone saves ~9 s). **~15 s (4%)**. Keeps coverage.
   Effort low; risk low.
5. **Library: reverse-graph step in `optimize` for host-resident graphs.** It currently does, for each column, an OpenMP
   gather, H2D copy, kernel launch and `sync_stream` (`cpp/src/neighbors/detail/cagra/graph_core.cuh:836-852`). Copy the
   graph to device once and reuse the device path at :828-834. **~13 s (3.6%)** on its own, ~7 s after idea 1. Effort
   low, risk low.
6. **Library: allocate the kvikio_ofstream staging buffer uninitialized or lazily.** `std::vector<char>` zero-fills
   32 MiB on every `cagra::serialize` (`cpp/src/util/file_io.cpp:265, 403`; default size at
   `include/cuvs/util/file_io.hpp:466`). Use `make_unique_for_overwrite<char[]>`, or cap the buffer at the bytes actually
   written. **~10 s (2.7%)** on its own, ~3 s after 1b. Effort low, risk low.
7. **Coverage-reducing options (only if 1-3 are not enough):**
   (a) FilterTest MULTI_KERNEL with unaligned dims (.cuh:2014-2030) runs `itopk_size` 256, which means ~260 iterations
   per query. Use 64, or 10 queries: ~10 s.
   (b) Keep IVF_PQ only for SINGLE_CTA in the corner-case block: ~33 s. Idea 1e gets the same saving without losing
   coverage.
   (c) Lower NN-descent `max_iterations` for tiny datasets (`cpp/src/neighbors/detail/nn_descent.cuh:2748-2789`): a few s.

Not worth doing: moving the test reference to the GPU or caching it. naive_knn, eval_distances and the recall loops
together cost ~4.5 s (1.2%).

Caution: keep this executable in one process. Each extra process pays ~7 s of JIT linking (per-process cache), ~1 s of
`CUFileInit`, and context start-up.

## Uncertainties

* Phase attribution is inferred, because CPU sampling was unavailable. Host time between two CUDA calls is charged to the
  phase of the preceding call; only JIT and kvikio spans were carved out. Bucket edges such as optimize vs. serialize
  are approximate (about ±10%).
* nsys/CUPTI inflate per-launch cost, and there are 11.2 M launches. Absolute seconds are therefore higher than in a
  plain run. The ctest `-j8` figure (313 s) also includes GPU sharing. The percentages should transfer roughly.
* Savings for the test-side ideas come from measured per-case times, minus the search, reference and JIT time that would
  remain. The ideas overlap and are not additive, so a combined figure is given instead.
* JIT (32.5 s here) and `CUFileInit` (8 x ~1 s) are inflated because every group ran in its own process. In a single
  ctest process the JIT cost is estimated, not measured, at 3-7 s. It is not counted in any savings above.
* The 15.5 ms gap during serialize is attributed to zero-filling the 32 MiB buffer. This is inferred: the main thread has
  no traced calls in that window and it sits right after the kvikio FileHandle construction, which matches the
  member-initializer order.
* The library savings in ideas 2, 3, 5 and 6 assume that the fixed code runs at the cost observed for the aligned /
  device-path equivalents. None were prototyped (no GPU runs or builds were allowed).
* The build-reuse ideas assume that reusing an index across search variants is acceptable for coverage. Search is const.
  Repeated identical builds would only catch nondeterministic build failures.
