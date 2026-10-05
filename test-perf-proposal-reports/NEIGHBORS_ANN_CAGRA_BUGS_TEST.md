# NEIGHBORS_ANN_CAGRA_BUGS_TEST: where the time goes and how to make it faster

Source: 15 `nsys profile -t cuda,nvtx,osrt` runs (one gtest filter per process; RTX 6000 Ada;
`staging-test-optimizations-local`; async RMM resource). The "nsys" column is gtest time with a warm JIT cache. The
"ctest" column comes from the `ctest -j8` run of the same binary (`Testing/Temporary/LastTest.log`, 2026-10-02 23:32):
one process, cold JIT cache right after a rebuild. Files are `cpp/tests/neighbors/ann_cagra/bug_*.cu`.

## Summary

| reproducer (file) | original failure condition (from source/commit) | cases | nsys s | ctest s | dominant cost |
|---|---|---|---|---|---|
| AnnCagraBugMultiCTACrash (`bug_multi_cta_crash.cu`) | #438: MULTI_CTA illegal address. With every seed distance inf, `compute_distance_to_random_nodes` left the best index uninitialised. Needs queries = `upper_bound<half>` and a fresh `raft::resources` for search | 1 | 9.82 | 10.42 | NN-descent build of 1,183,514 x 100 half (8.9 s) |
| cagra_extreme_inputs_oob_test (`bug_extreme_inputs_oob.cu`) | #337/#460, #565: IVF-PQ build + refine (rate 2) on N(0, 1e20) data gives invalid candidate ids. The build must throw, not go OOB in refine / `kern_prune` | 1 | 3.30 | 3.80 | refine 1.95 s, IVF-PQ training 0.95 s |
| CagraIterativeBuildBugTest {f32,i8,u8} x {dense,VPQ} (`bug_iterative_cagra_build.cu`) | #1818: with `graph_degree == intermediate_graph_degree`, the iterative build writes degree+1 columns into a degree-wide buffer | 6 | 1.17 | 20.34 | cold JIT: 4 stalls of ~5 s |
| Issue93Reproducer (`bug_issue_93_reproducer.cu`) | cuvs-lucene#93: racing `cudaFuncSetAttribute(smem)` from 4 threads, x50 rounds | 1 | 1.71 | 0.07 | nsys artefact (see below) |
| cagra_hashmap_bitlen_no_hang_test (`bug_issue_2523_hashmap_bitlen.cu`) | #2523: hash bit-length loop hangs for oversized itopk | 4 | 1.43 | 0.32 | same 1000x32 index built in each SetUp |
| cagra_nan_queries_test (`bug_nan_queries_multi_kernel.cu`) | NaN queries leave the seed index uninitialised, then the graph is read OOB | 3 | 1.02 | 0.39 | same 10000x32 index built 3 times |
| cagra_graph_smaller_than_dataset_test | seeds must use `graph.extent(0)`, not the dataset size | 1 | 0.48 | 0.25 | 2 small builds |
| **Total** | | **17** | **18.9** | **35.6** | warm single process: ~15.4 s |

## Where the time goes

* **Per-process start-up is an artefact of profiling one group per process.** Each group spends ~0.5 s before its first
  CUDA call, ~0.12 s creating the pool and some time on first JIT loads; nsys adds ~3 s of wall time (8 s for g000).
  **Issue93's 1.7 s is a tracing artefact.** Each of 200 short-lived threads blocks 10-40 ms on its first CUDA call,
  serialised behind a lock (cudaMallocFromPoolAsync p90 19.7 ms, median 4 us). Untraced the test takes 71 ms.
* **Cold JIT: 20.2 s (57%) of the ctest run.** Four iterative cases (f32 dense, f32 VPQ, i8 dense, u8 dense) each stalled
  4.93-5.15 s between the log lines "Current graph size 625" and "1250", i.e. during the first iterative CAGRA search.
  A pair of `~/.nv/ComputeCache` entries (~1.4 MB + ~0.8 MB) was written at 23:32:14.53, 19.72, 24.71 and 29.68, the
  exact end of each stall. With a warm cache this step takes 22-40 ms. `n_dim = 1024` picks the 512-wide / team-32
  descriptor (`compute_distance_standard_matrix.json`), which no other test here uses. The small-dim tests (128-wide)
  took 49-251 ms in the same run, with no stall.
* **MultiCTA (9.8 s): the search under test takes 6 ms.** NN-descent runs 0.95-9.84 s: 20 iterations of ~0.42 s. Of that,
  local_join takes 2.05 s and add_rev_edges 0.82 s of GPU time; D2H copies take 1.86 s (11 GiB); the rest is host time,
  including a 0.69 s tail. Optimize takes 0.35 s and update_dataset + search 0.12 s. 1,183,514 x 100 is the GloVe-100
  shape, apparently copied from the report; the PR names no size-related condition.
* **Extreme inputs (3.3 s).** `ivf_pq_params{}` keeps the library defaults (n_lists 1024, 20 iterations, batch 4096).
  Coarse k-means and rotation take 0.64-0.76 s; per-subspace PQ codebooks 0.76-1.56 s (3.7k `fused_1nn` launches).
  Then 25 batches of IVF-PQ search + device refine take 1.6-3.55 s (~80 ms each, GPU 28% busy):
  `fill_refinement_index` resizes one IVF-Flat list per query (`cpp/src/neighbors/ivf_flat/ivf_flat_build.cuh:489-491`),
  which means 237k pool allocations, 202k H2D copies and 109k map_kernel launches. Finally `kern_fused_prune` throws
  the expected error ("...invalid or duplicated neighbor nodes...", `graph_core.cuh:1530`).
* **The other 11 cases take ~1.2 s together (ctest):** 1k-10k-row builds and tiny searches.

## Ideas (ranked by estimated savings; % of the 35.6 s ctest run / of the ~15.4 s warm total)

1. **MultiCTA: give the index a synthetic graph instead of building one.** Keeps coverage of the bug. In place of
   `cagra::build` + `update_dataset` (`bug_multi_cta_crash.cu:28-35`), fill a member graph [n_samples, 32] with
   `raft::random::uniformInt` in [0, n_samples) and pass it to `cagra::device_padded_index<half>(res, metric,
   build_padded_->view, graph)` (`cagra.hpp:628`). Keep the dataset, the inf queries, the search params and `res_search`.
   Saves **~9.5-10 s (28% / 63%)**. Effort low.
   Does it still trigger #438? Yes. Every query-node distance is inf whatever the graph is, so the uninitialised seed
   condition is unchanged. What is lost is incidental coverage of a 1.18M-row half NN-descent build and optimize.
   Shrinking `n_samples` (:89) instead would save less and change the seed range. Not preferred.
2. **Iterative build: set `n_dim` from 1024 to 128** (`bug_iterative_cagra_build.cu:112`; `pq_dim = n_dim/4` at :65 stays
   valid). Saves **up to ~20 s on a cold cache (57%)**, ~0 when warm. Effort trivial.
   Does it still trigger #1818? Yes: dim plays no part (commit 6026d4db). **Do not shrink `n_samples`**: the overflow
   needs a search chunk with rows x 17 > 8192 x 16, i.e. more than 7,710 rows, and 10,000 rows gives an 8,192-row chunk.
   Partially reduces coverage: the 512-wide descriptor is no longer exercised by the iterative build.
   Environment alternative that keeps dim 1024: persist `~/.nv/ComputeCache` in CI or raise `CUDA_CACHE_MAXSIZE`.
   The local cache is 1.1 GB, probably at its cap, so it may be evicting entries.
3. **Extreme inputs: smaller input and an exact assertion.** Set `n_samples` from 100000 to 10000
   (`bug_extreme_inputs_oob.cu:64`; refine cost is proportional to rows): ~1.7 s. Optionally also set `n_dim` from 200
   to 32 (:65) or reduce `build_params.n_lists` (:26-31) to cut PQ training: ~0.7 s. Total **1.7-2.5 s (5-7% / 11-16%)**.
   Risk medium. The inf/NaN distances come from the 1e20 scale (squares overflow float at any dim), so the condition
   should persist. But `catch (const std::exception&)` (:38) accepts *any* error, so assert the message too, and confirm
   once on a GPU that the throw still comes from `graph_core.cuh:1530`. Keeps coverage once that is verified.
4. **Library: allocate the refinement lists once.** In `fill_refinement_index` (`ivf_flat_build.cuh:489-491`), use one
   contiguous allocation instead of one per query. Saves ~1.7 s here (overlaps idea 3) and speeds up every device
   `refine`, including CAGRA IVF-PQ builds with `refinement_rate > 1`. Effort medium, risk low. Keeps coverage.
5. **Build each index once per suite.** Move the hashmap build to `SetUpTestSuite` (`bug_issue_2523_hashmap_bitlen.cu:54-74`),
   and build the NaN-queries index once for all 3 algos (`bug_nan_queries_multi_kernel.cu:46-51, 81-85`). Saves ~0.35 s (1%).
   Effort and risk low. Keeps coverage.

Combined, 1+2+3+5 cut the ctest time from 35.6 s to ~3-5 s (warm: ~15.4 s to ~3 s). Not worth doing: Issue93 (71 ms) and graph-smaller (0.25 s).

## Uncertainties

* The 5 s stalls are attributed to JIT compilation from the timing of the cache writes; nothing was traced cold. It is
  unproven that the 128-wide variants compile quickly when cold: they only showed no stall in that run (a concurrent test
  may have compiled them first). Whether CI pays the 20 s depends on cache persistence.
* Phases come from kernel names (CPU sampling was unavailable). nsys inflates per-call costs (g000 has ~600k traced
  calls). No savings were measured, because no GPU runs or builds were allowed.
* #438 was intermittent and easier to hit without a pool allocator (per the PR). The test now uses the async pool and
  checks no outputs, so it already detects regressions weakly; idea 1 does not change that. An id check like the one in
  the NaN-queries test (each id < n_samples or the invalid sentinel) would help. #1818's write lands 32 KB past a pool
  block, so it is only reliably caught under compute-sanitizer.
