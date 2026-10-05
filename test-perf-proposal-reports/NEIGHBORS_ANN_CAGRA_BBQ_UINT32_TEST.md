# NEIGHBORS_ANN_CAGRA_BBQ_UINT32_TEST: where the time goes and how to make it faster

Source: 5 `nsys profile -t cuda,nvtx,osrt` runs, one TEST_P per process, RTX 6000 Ada, branch `staging-test-optimizations-local`. The gtest sum under nsys is **16.4 s** (wall 35.9 s; ~4 s per process is start-up plus nsys). An earlier plain single-process run took 14.0 s, and `ctest -j8` took 23 s.
Tests: `cpp/tests/neighbors/ann_cagra/test_bbq_uint32_t.cu` (5 TEST_Ps x 27 params). Fixture: `cpp/tests/neighbors/ann_cagra_bbq.cuh` (`.cuh` below). Each param has n_rows 4000, dim 128, degree 32 (intermediate 64), 200 queries; 9 code layouts x 3 metrics (L2, IP, Cosine). General CAGRA build and search costs are covered in `NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST.md`.

## Summary

| TEST_P | cases | gtest s (nsys) | plain run s | GPU busy | kernels | work per case |
|---|---|---|---|---|---|---|
| AnnCagraBbqSearchRecall | 27 | 5.9 | 6.3 | 16% | 9198 | dense NN-descent build + BBQ build, 2 optimizes, naive GT, 2 searches |
| AnnCagraBbqGraphShape | 27 | 4.3 | 2.7 | 12% | 4473 | 1 BBQ build + host range check (this nsys run had host contention, see below) |
| AnnCagraBbqSerializeRoundTrip | 27 | 3.2 | 2.4 | 17% | 4635 | 1 BBQ build + serialize/deserialize + 2 SINGLE_CTA searches |
| AnnCagraBbqGraphOnlyBuild | 27 | 2.8 | 2.5 | 19% | 4473 | 1 BBQ build (`attach_dataset_on_build=false`) |
| AnnCagraBbqUnsupportedParams | 27 | 0.1 | 0.1 | 8% | 126 | quantize + 3 expected throws (~1 ms) |
| **Total** | **135** | **16.4** | **14.0** | ~15% | 23 k | **135 NN-descent builds: 108 BBQ + 27 dense** |

## Where the time goes

Method: each case was cut at its two SetUp `GenerateRoundingErrorFreeDataset` launches. Main-thread time was then split into phases using CUDA API and OSRT markers: GNND pinned allocs, then the first `pthread_create` (start of the loop), the last `pthread_join`, `kern_sort*` (optimize) and `kern_merge_graph`. The phases add up to 16.3 s.

| bucket | s | % | detail |
|---|---|---|---|
| NN-descent iteration loop | 10.5 | 64 | **All 135 builds run exactly 20 iterations (2700 `local_join` launches).** The termination test (`nn_descent.cuh:2886`) never fires at this size. One iteration takes 3.2-5.6 ms, but `local_join` is only 0.3-0.4 ms of GPU time. The main thread spends 1.6-3.8 ms in `pthread_join`, waiting on the per-iteration `std::thread` that runs `update_graph` + `sample_graph` (OpenMP, :2277/:2232). It spends another 1.1-1.6 ms in `sample_graph_new` and copies. `pthread_join` alone totals 6.2 s. |
| NN-descent init + finish | 2.4 | 14 | 9-16 ms per build: about 9 `cuMemAllocHost`/`cuMemFreeHost` per build (0.49 s in total), `init_random_graph`, the final `update_graph`, `sort_lists` and `shrink_graph_removing_duplicates`. |
| CAGRA optimize | 1.8 | 11 | ~12 ms per build. 0.84 s of it is the host-graph reverse-edge loop (32 columns x gather + H2D + kernel + sync, `graph_core.cuh:836-852`; float doc idea 5). |
| update_dataset / serialize / checks / search setup | 0.7 | 4 | Includes ~0.13 s of first-search JIT for the first metric in each process. |
| CAGRA search (4 per param) | 0.4 | 2 | |
| quantize (D2H + load codes from the `/tmp/bbq-*.bin` cache + upload) | 0.4 | 2 | Cache was warm. |
| SetUp, naive GT, teardown | 0.2 | 1 | Not worth optimizing. |

* **Builds are redundant.** The 4 build-calling TEST_Ps build the same BBQ graph per param: same codes, same `default_index_params()` (.cuh:85-96).
  GraphOnlyBuild differs only in `attach_dataset_on_build`, which takes effect after the graph is built (`cagra_build.cuh:2986-2993`).
  So there are 108 BBQ builds for 27 distinct graphs. The dense reference build in SearchRecall (.cuh:175) depends only on the metric (data seed 1234), so it is 27 builds for 3 distinct graphs.
* **The loop is host-bound and sensitive to host load.** GraphShape and GraphOnlyBuild do identical builds, yet took 111 vs 64 ms of loop time per build
  (`pthread_join` 3.8 vs 1.6 ms per iteration) in runs only minutes apart. This fits two 36-thread OpenMP teams (main and helper) running tiny 4000-row loops on a shared host.
  It probably also explains 14 s in the plain run vs 23 s under `ctest -j8`.
* **Per-process artifacts** (not per-test work): ~4 s of start-up per nsys run, plus a 0.25-0.4 s slower first case per process (search JIT, OpenMP pool, pool growth). In a single ctest process the first case is ~0.5 s slower in total.

## Ideas (ranked by estimated savings; % of 16.4 s; plain-run figures in brackets)

1. **Build each BBQ graph once per param. Keeps coverage.** Move the GraphShape asserts (.cuh:217-232) and the serialize round trip (.cuh:255-274) into the
   SearchRecall BBQ block (.cuh:180-191), before `graph_index` is moved at :189. Alternatively keep the TEST_P names and share a per-param static cache
   of codes plus host graph, and build each test's index with the `index(res, metric, dataset, graph)` constructor, as the build itself does at cagra_build.cuh:2987.
   Removes 54 builds: **~7.0 s (43%)** [4.5 s, 32%]. Effort low-medium; risk low (fewer, coarser failure messages, or shared state with the cache).
2. **Library: make the NN-descent host loop cheap for small n.** Cap the OpenMP team for small `nrow` in the `GnndGraph` loops (`nn_descent.cuh:2164-2316`).
   Replace the `std::thread` per iteration (:2900) with one persistent worker. Drop the per-row `std::vector` in `sort_lists` (:2300).
   Today the host spends ~3.9 ms per iteration against 0.4 ms of GPU work. Bringing that to ~1 ms would save **~5-8 s (30-48%) now, ~1.5 s after 1+4+5**, and less inflation under ctest -j8.
   Keeps coverage. Effort medium; risk medium. The savings are a guess: it could not be confirmed what the host time is (no CPU sampling).
3. **Fewer NN-descent iterations in the test**: set `max_iterations` to ~10 on the `nn_descent_params` built at .cuh:93-94. Both the dense and BBQ builds get it, so the ratio check stays fair.
   **~5.3 s (32%) alone, ~1.1 s after 1+4+5.** Effort low. Risk medium: lower graph quality, so `min_recall_ratio` and the `> 0.8` baseline (.cuh:205) must be
   re-validated on a GPU. It keeps code coverage but weakens the quality check.
4. **Compute the dense reference recall once per metric. Keeps coverage.** Use a static `metric -> reference_recall` (or ground truth + dense graph) cache in
   SearchRecall (.cuh:173-178). Removes 24 dense builds: **~2.3 s (14%)** [2.4 s]. Effort low; risk low (NN-descent jitter in the reference is the same as today).
5. **GraphOnlyBuild without its own build per param.** (a) Keeps coverage: in the merged test from idea 1, build with `attach_dataset_on_build=false`,
   run the graph-only asserts (.cuh:286-289), then attach the codes through the index constructor for the GraphShape asserts. (b) Reduces coverage
   slightly: run GraphOnlyBuild on 1-3 params, since the flag does not depend on layout or metric. **~2.3 s (14%)** [2.2 s]. Effort low; risk low.
6. **Library: optimize reverse graph from a device copy** (`graph_core.cuh:836-852`, float doc idea 5). **~0.6 s (4%)** now, ~0.15 s after 1+4+5. Keeps coverage. Effort low; risk low.
7. **Library: pool the pinned NN-descent buffers** (GNND/GnndGraph constructors, ~9 `cuMemAllocHost` per build). **~0.4 s (3%)**, ~0.1 s after 1+4+5. Keeps coverage. Effort medium; risk low.
8. **Reduces coverage:** run the 9 code specs for L2 only, and IP/Cosine for 3 representative layouts (.cuh:369-372). 27 -> 15 params, ~1.2 s more after 1+4+5.

Combined, test-only and coverage-keeping (1+4+5): 135 -> 30 builds, **~11.5 s (70%)**, leaving ~4.7 s [plain: ~9 s saved, 14.0 -> ~5 s]. Adding 2 or 3 brings it to ~3.5 s. Keep the executable in one process: each extra process pays start-up plus ~0.3 s of first-case warm-up.

## Uncertainties

* No CPU sampling (perf_event_paranoid=4). Host phases are inferred from gaps between CUDA/OSRT calls. It is not known how the ~3.9 ms of host time per iteration
  splits between real work, OpenMP fork/join and host contention, so idea 2 is the least certain.
* Run-to-run variance is large because the loop is host-bound (identical builds took 64-111 ms of loop time). Per-TEST_P nsys numbers carry about ±40% noise.
  The plain single-process run (14.0 s, cold BBQ cache) is given as a cross-check.
* The profiles ran with a warm `/tmp/bbq-n4000-*.bin` cache (the CPU quantizer caches codes by n_rows/dim/layout/metric; the key has no data seed).
  In a fresh CI container the first case per (layout, metric) quantizes on the CPU. The cold run shows +5-60 ms each, 18 files: ≤0.5 s, paid once.
* Savings assume ~0.1 s per build (nsys) and that merged tests keep the same asserts. The ideas overlap; the combined figure accounts for that.
* Idea 3 and the BBQ recall thresholds were not run (no GPU work was allowed). nsys/CUPTI inflate per-call costs a little (23 k launches), so percentages transfer better than absolute seconds.
