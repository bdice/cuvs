# NEIGHBORS_ANN_IVF_RABITQ_TEST: where the time goes and how to make it faster

Source: 5 nsys runs, one per TEST_P (`-t cuda,nvtx,osrt`, RTX 6000 Ada). All times are gtest time under nsys. The
whole executable took **55 s** under `ctest -j8`, about the same as the 58 s summed here. The dominant cost is
single-threaded host CPU work, which tracing does not inflate, so the nsys seconds are close to real seconds.

## Summary

| metric | value |
|---|---|
| cases / groups | 230 / 5: the same 46 parameter sets × 5 build paths `build_search`, `build_host_input_search`, `build_serialize_search`, `build_host_input_serialize_search`, `build_forced_streaming` (`ann_ivf_rabitq/test_float_int64_t.cu:12-19`) |
| total gtest time (nsys) | **58.0 s** (host_input_serialize 13.5, serialize 12.4, forced_streaming 11.5, host_input 10.9, build_search 9.7) |
| GPU busy (kernels + copies + memsets) | 3.6 s = **5–7 %** of the CUDA span. Kernels total 2.7 s. The test is host-bound. |
| kernel launches | 404 k (1.6–1.9 k per case, mean 6.8 µs) |
| one case dominates | `*/28` `{num_db_vecs=32768, num_queries=64, k=16384}`: 7.9–8.4 s in every group, **40.8 s = 70 %** |
| the other cases | 225 cases at 20–90 ms each (13.4 s), plus the first case of each process (3.8 s) |

## Where the time goes

Each case was split using its `rngKernel` markers: 5 per case (7 in the serialize groups). `gen_data` produces 2 of them,
`sample_rows` at the start of the build 1, and the rotator 2. The construct phase ends at the last `exrabitq_fused_kernel_batch`.

| phase (230 cases) | s | % | code |
|---|---|---|---|
| **host recall check of case 28 (×5)** | **40.45** | **69.8** | `calc_recall` (`ann_utils.cuh:228-279`) ≈ 7.7 s per case + `check_unique_indices` (`:163-193`) ≈ 0.2 s, called from `ann_ivf_rabitq.cuh:217` |
| balanced k-means in `build` | 7.0 | 12.1 | `ivf_rabitq.cu:142-166` → `kmeans_balanced.cuh` `build_hierarchical` (`:1292`) |
| search, JIT link, (de)serialize (cases 1–45) | 3.7 | 6.3 | serialize ≈ 2.2 s, JIT ≈ 1.0 s, search ≈ 0.5 s |
| first case of each process | 3.8 | 6.5 | module load and handle init; 2 × `CUFileInit` (0.95 s each) in the serialize groups |
| index construct | 2.6 | 4.6 | `RotatorGPU` QR (`rotator_gpu.cu:24-30`) + per-list quantize loop (`ivf_gpu.cu:526-531`, 32 × 5 launches) |
| `gen_data` + `naive_knn` + host copies | 0.9 | 1.5 | `ann_ivf_rabitq.cuh:83-124` |

* **Case 28 is a CPU loop.** Its GPU work is only ≈ 40 ms (naive 7, k-means 18, construct 4, search + JIT 12 ms).
  After that, the GPU sits idle for 7.9–8.3 s (`map_kernel → rngKernel`). The log confirms it: the
  `Recall = 0.733 (768534/1048576)` line, printed after `calc_recall` returns, appears 7.73 s after the previous case's
  line. The next case starts 0.2 s later (64 `std::set`s of 16384 inserts). `calc_recall` scans the expected row
  linearly for every actual neighbour, **twice** (distance-aware pass + index pass). At 27 % misses that is
  ≈ 2 × 1.1·10¹⁰ iterations. All 5 copies of the case run the same search (`k > kMaxTopKBlockSort = 64`,
  `searcher_gpu.cuh:26`, selects the non-block-sort + `select_k` path).
* **k-means: the same launch-bound pattern as IVF-PQ (idea E) and IVF-Flat (idea C).** `n_lists = 32` is set in the
  `ivf_rabitq_inputs` constructor (`ann_ivf_rabitq.cuh:27`, from the default 4096 rows, so case 28 also gets 32 lists).
  `build_hierarchical` therefore trains 6 mesoclusters, then 6 fine sets × 20 iterations, then 2 final iterations:
  **≈ 144 EM iterations per build** (counted from `fused_1nn` launches). Each iteration costs 133 µs of wall time in
  `build_search` and ≈ 237 µs in the host-input groups, vs ≈ 50 µs of GPU time, with ≈ 0.95 blocking D2H copies
  (`adjust_centers`). The kernel sequences are identical; the gap between groups is contention (launch latency 3.6 vs 5.7–7.8 µs).
* **Serialization adds 24–26 ms per case** (2.2 s in total), plus `CUFileInit` 0.95 s once per process. The kvikio writes take < 1 ms.
  **10–15 ms** (median 10.2 / 15.2 ms) is untraced main-thread time right after the file is opened.
  `kvikio_ofstream` allocates a **32 MiB `std::vector<char>` staging buffer, zero-filled on every save** (`file_io.cpp:265`,
  member at `:403`; default size `file_io.hpp:466`). Freeing it costs another ≈ 2 ms. Deserialize also builds two throwaway
  random rotators before the saved one overwrites them: a 128-dim one in `IVFGPU(handle)` (`ivf_gpu.cuh:161`), and
  `RotatorGPU(handle_, dim)` in `load` (`ivf_gpu.cu:145`). Each is a QR: ≈ 1 ms, or ≈ 40 ms at dim 2048.
* **Per-process costs.** Each process JIT-links 18 search-kernel configurations, at ≈ 11–13 ms of host time each before
  `cudaLibraryLoadData` (0.16–0.27 s per process). The first case takes 0.3–0.4 s, or 1.35 s with `CUFileInit`. A single-process
  ctest run pays these once (≈ 1.5 s), not 5×.
* **Redundancy.** Each group builds only 26 distinct indices. 20 of the 46 cases rebuild an index already built in the same
  group (`n_probes` ×6, `k ≤ 64` ×5, search modes ×4, 1-bit modes ×4, `bits_per_dim = 3`). Their k-means + construct
  (+ serialize) costs 3.8 s. **Exact duplicates:** cases 25 (`k = 10`), 31 (`bits_per_dim = 3`) and 40 (`QUANT4`, the default)
  equal case 0, and 44 equals 29. That is 20 cases, 0.83 s. Search-only parameter sets are crossed with all 5 build paths
  (3.9 s outside `build_search`). Big dims cost 4.75 s (8 %). 2049 and 2050 both pad to 2112 (`ivf_gpu.cu:46`).
* **Not hotspots:** `naive_knn` and data generation (0.9 s), the search kernels (< 1 ms per case), forced streaming (it costs
  the same as host input), and process start (≈ 0.6 s).

## Ideas (ranked by standalone savings; % of the 58 s)

**A. Make the host recall check O(k log k)** *(keeps coverage)*
* Change: in `calc_recall` (`ann_utils.cuh:228-279`), sort per-row copies of the expected indices and distances once.
  Find index matches by binary search. Find distance matches with `lower_bound`, checking the nearest expected value
  on each side (the `CompareApprox` ratio is monotone in |a−b| for distances ≥ 0, so the result is identical).
  Compute both counts in one pass. In `check_unique_indices` (`:163-193`), use sort + `adjacent_find` instead of `std::set`.
* Savings: **≈ 39.5 s (68 %; ≈ 39 s real, pure CPU)**. Effort: S. Risk: L. This is the same change as IVF-PQ H and
  IVF-Flat H, and every ANN test that uses `eval_neighbours` benefits.
* Test-only alternative *(less very-large-k coverage)*: `16384 → 2048` in `var_k()` (`ann_ivf_rabitq.cuh:308`).
  This keeps the `k > 64` path and saves ≈ 38 s. Running k = 16384 only in `build_search` saves ≈ 31.6 s.

**B. Use fewer k-means iterations in the tests** *(keeps code paths, changes the configuration)*
* Change: `index_params.kmeans_n_iters = 5` in the inputs constructor (`ann_ivf_rabitq.cuh:27`): ≈ 144 → ≈ 40 EM
  iterations per build.
* Savings: ≈ 5 s (9 %). Effort: XS. Risk: **M**. Re-validate on the GPU: `n_probes` 1/2/4 have recall 0.15/0.25/0.40 vs
  `min_recall` 0.08/0.16/0.32, and the 1-bit cases have 0.35–0.36 vs 0.30.

**C. Cache the built index across cases with the same build key** *(keeps coverage)*
* Change: in `run()` (`ann_ivf_rabitq.cuh:185`), keep a static cache keyed by (TEST_P, `num_db_vecs`, `dim`, `index_params`),
  as in IVF-PQ idea B.
* Savings: 3.8 s (6.6 %), ≈ 1.3 s after B. Effort: M. Risk: L–M. `search` takes a non-const `index&`
  (`ivf_rabitq.cu:191`), so first confirm that it does not mutate the index.

**D. Cut per-iteration overhead in the library's k-means** *(keeps coverage)*: the IVF-PQ idea E changes
(`adjust_centers` copy + sort, norms, workspaces). 35–50 % of k-means ≈ 2.5–3.5 s (4–6 %), less after B. Effort: M. Risk: L–M.

**E. Don't cross search-only parameters with the 4 other build paths** *(reduces combination coverage)*: give `n_probes`,
`k ≤ 64` and the search modes their own fixture alias, instantiated only for `build_search` (`test_float_int64_t.cu:10-19`).
3.9 s (6.8 %). This overlaps C, so pick one. Effort: S. Risk: L.

**F. Stop zero-filling the `kvikio_ofstream` staging buffer** *(keeps coverage, library)*: allocate it uninitialised
(`make_unique_for_overwrite<char[]>`) or grow it lazily up to the cap (`file_io.cpp:265,403`). ≈ 1.2 s here (2 %); every
`kvikio_ofstream` save benefits too (IVF-PQ, IVF-SQ, CAGRA, Vamana, ScaNN, brute force). Effort: S. Risk: L.

**G. Delete the exact duplicates** *(keeps coverage)*: drop `k = 10` (`:308`), `bits_per_dim = 3` (`:321`) and `QUANT4` from
both mode lists (`:335-338`, `:350-353`). 20 cases, 0.83 s (1.4 %). Effort: XS.

**H. Drop dim 2050 from `big_dims()`** (`:287`) *(reduces coverage)*: it has the same padded dim as 2049. 0.87 s (1.5 %). Effort: XS.

**I. Minor library fixes** *(keep coverage; each < 1 %)*: construct the deserialization index without random rotators
(`ivf_gpu.cuh:161`, `ivf_gpu.cu:145`), ≈ 0.3 s; batch the per-cluster quantize loop (`ivf_gpu.cu:526-531`), ≈ 0.3 s.

The recommended combination that keeps coverage is **A + B + G + F: ≈ 46 s (≈ 79 %)**, all with XS–S effort. After that, apply C or D.

## Uncertainties

* **Attribution without CPU sampling.** The 7.7 s / 0.2 s split between `calc_recall` and `check_unique_indices` comes from where
  the Recall log line falls. The staging-buffer cost is inferred from the untraced main-thread gap after the file `open`, plus the code.
* **Profiling noise.** Other jobs were profiling on the same GPU and host. Launch latency differed up to 2× between groups, so
  the k-means and launch-bound numbers carry about ±40 % noise. The 40 s CPU loop is unaffected.
* **Phase boundaries are inferred** from kernel markers (no cuVS NVTX ranges); they are accurate to about 1 ms per case.
* **Not validated on the GPU:** recall with `kmeans_n_iters = 5` (B), the cache (C), and the library estimates (D, F, I).
