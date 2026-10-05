# NEIGHBORS_ANN_CAGRA_INT8_UINT32_TEST: where the time goes and how to make it faster

Source: 12 `nsys profile -t cuda,nvtx,osrt` runs (one fixture group or gtest shard per process), RTX 6000 Ada, branch
`staging-test-optimizations-local` (#2685, #2726). gtest times under nsys sum to 208.4 s (251 s under `ctest -j8`).
`.cu` = `cpp/tests/neighbors/ann_cagra/test_int8_t_uint32_t.cu`, `.cuh` = `cpp/tests/neighbors/ann_cagra.cuh`. The shared
findings are explained in [the float doc](NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST.md) and only quantified for int8 here.
How int8 differs from float:
* **One TEST_P per fixture.** There is no `_U32`/`_I64` doubling, no FilteredMerge and no CagraQ (`.cu:12-39`), so
  float-doc ideas 1a/1b do not apply.
* **More dims take the one-query-at-a-time search fallback.** A batch needs a 16 B query row pitch
  (`cagra_search.cuh:94-114`). For int8 that means `dim % 16 == 0`, so dim 8 is unaligned too. This is why FilterTest
  costs 36.0 s here against 22.7 s for float.
* **Data:** `uniformInt` in [-10, 10]. For InnerProduct, rows are rescaled to a norm of 20·sqrt(dim) and clamped
  (`.cuh:237-275`). There is no dim-based skip, and setup costs 1.2 s in total. BitwiseHamming is always skipped:
  382 of the 1128 cases are skipped, at 0.3 s.

**int8 vs uint8** ([uint8 doc](NEIGHBORS_ANN_CAGRA_UINT8_UINT32_TEST.md)): the inputs, the 16 B rule and the totals are the
same (208.4 vs 207.8 s); FilterTest is 36.0 s in both. uint8 also runs 61 Hamming cases (~7 s), so its AnnCagraTest
(56.2 s) and AddNodes (12.1 s) are larger. int8 spends more time in NN-descent host code: MultiPartition takes 24.4 s
(uint8 20.6 s), and IndexMerge InnerProduct/Cosine PHYSICAL NN-descent cases cost +3.9 s over float.

## Summary

| TEST_P (`.cu` line) | cases (skipped) | gtest s | % | GPU busy | kernel launches | dominant cost |
|---|---|---|---|---|---|---|
| AnnCagraIndexMergeTest (:19) | 459 (197) | 84.0 | 40 | 17% | 2.60 M | IVF-PQ graph build 46.5 s (2-3 builds per case) |
| AnnCagraTest (:13) | 459 (161) | 53.1 | 25 | 13% | 1.15 M | IVF-PQ 21.3 s; 108 indistinguishable duplicate cases (18.0 s) |
| AnnCagraFilterTest (:17) | 60 (0) | 36.0 | 17 | 42% | 3.36 M | one-query-at-a-time search 30.6 s (MULTI_KERNEL 24.8 s) |
| AnnCagraMultiPartitionTest Search + FilteredSearch (:34-35) | 24 + 24 | 14.1 + 10.3 | 12 | 12% | 0.05 M | NN-descent 16.8 s; 48 builds for 13 partition sets |
| AnnCagraAddNodesTest (:15) | 102 (24) | 10.9 | 5 | 81%* | 0.13 M | iterative-build searches one query at a time: 7.4 s |
| **Total** | **1128 (382)** | **208.4** | | **23%** (48 of 210 s) | **7.29 M** | host / launch bound |

\*GPU busy is high because of back-to-back 1-block `search_single_cta` launches (7.3 s), one per query.

## Where the time goes

Method: each case was cut out of the timeline using the two int8 `uniformInt` `rngKernel` launches in SetUp. There are
exactly two per case, including skipped cases, and the phase sum (208.2 s) matches gtest. Main-thread time was split by
the family of the last launched kernel. JIT (the host gap before `cudaLibraryLoadData`) and the serialize window (the host
gap before kvikio's `pwrite`, through the last `pread`) were carved out separately.

| bucket | s | % | detail |
|---|---|---|---|
| IVF-PQ graph build (k-means + PQ codebooks, plus 0.6 s of IVF-PQ kNN search) | 68.4 | 32.8 | IndexMerge 46.8, AnnCagraTest 21.6. Each build costs ~3.7k launches and 308 syncs, with a floor of ~100-150 ms even for 3-row datasets. The corner-case block (`.cuh:1805-1826`, n_rows ≤ 101) costs 21.7 s of it (120 cases); dim 1024 costs 0.74 s per build. |
| CAGRA search (the test's own searches) | 43.4 | 20.8 | **FilterTest 30.6 s**: 16 MULTI_KERNEL cases with `n_queries=100` at unaligned dims, 826k iterations x (4 kernels + blocking D2H `get_value` + sync), ~30 µs each. IndexMerge 7.9, AnnCagraTest 3.1, AddNodes 1.6. |
| NN-descent graph build | 35.4 | 17.0 | MultiPartition 16.8, IndexMerge 10.7. The main thread waits **20.6 s in `pthread_join`** for the per-iteration `update_and_sample` host thread (`nn_descent.cuh:2754-2789`; OpenMP `update_graph` at :2277), ~1.8 ms per iteration across ~11.7k iterations. GPU `local_join` is only 2.0 s. Every build runs all 20 iterations. |
| CAGRA graph optimize | 19.7 | 9.5 | Includes the per-column reverse-graph loop: 9.1 s, 1849 runs x ~4.9 ms (`graph_core.cuh:837-852`). |
| JIT link | 15.5 | 7.4 | 951 loads at ~16 ms each across 12 processes. The first case in each AnnCagraTest shard takes 1.6 s, of which `CUFileInit` is ~0.7 s. Both are mostly artifacts of one process per group. |
| searches inside the iterative build | 13.0 | 6.2 | All of it comes from unaligned dims (same fallback). AddNodes 7.4, IndexMerge 3.2, AnnCagraTest 2.4. |
| serialize + deserialize (AnnCagraTest) | 8.4 | 4.0 | 258 cases have a ~17 ms host-only gap before the first `pwrite` (the 32 MiB `kvikio_ofstream` zero-fill): ~4.3 s. |
| reference, verification, data generation | 4.5 | 2.2 | `naive_knn` 2.4, `eval_distances` 0.8, setup 1.2. **Not worth optimizing.** |

**int8-specific cost of the dim-8 fallback: ~16 s (7.7%)**, measured against the same cases in the float run, where
dim 8 batches. FilterTest's 33 dim-8 cases cost 16.5 s (float 3.5 s), AddNodes' 9 dim-8 cases cost 2.1 s (float 0.1 s),
and the other fixtures add ~0.8 s. The fallback across all unaligned dims (1, 3, 7, 8, 10, 17, 102, 137) costs **~51 s**:
FilterTest ~29, IndexMerge ~9, AddNodes ~8.6, AnnCagraTest ~4.5.

## Ideas (ranked by estimated savings; % of the 208 s; the ideas overlap, so a combined figure is given at the end)

1. **Test: build each index once and reuse it (float-doc idea 1, uint8-doc idea 1). Keeps coverage.**
   a. *Drop the 108 AnnCagraTest cases that differ only in `merge_strategy`, `host_dataset`, `itopk_size` or `search_width`.*
      AnnCagraTest ignores these axes. Filter `inputs` at `.cu:21`, or give AnnCagraTest its own list.
      **18.0 s (8.6%)**. Low effort, no risk.
   b. *IndexMerge: build the two halves once and merge them both PHYSICAL and LOGICAL.* **~14 s (6.6%)**. Low effort.
   c. *Process-wide index cache* (AnnCagraTest, IndexMerge, FilterTest). There are 298 / 262 / 60 executed builds but only
      107 / 101 / 13 distinct keys. **~54-68 s (26-33%) including a and b.** The low end keeps merged results per strategy.
      Medium effort. Risk: shared state.
2. **Library: batch queries when the query stride != dim (float-doc idea 3). Keeps coverage.** Pass a query leading
   dimension into `setup_workspace` (`jit_lto_kernels/setup_workspace_impl.cuh:53,143`) and into the multi-kernel kernels,
   then drop the per-query loop (`cagra_search.cuh:114-160`). **~51 s (24%)**, of which ~16 s is int8-only (dim 8).
   Medium effort and risk. *Cheaper partial fix:* check `terminate_flag` every N iterations instead of doing a D2H plus
   sync on every one (`search_multi_kernel.cuh:587-592`). The 839k blocking copies take 12.6 s here: **~11 s (5%)**,
   low effort.
3. **Library: train the IVF-PQ codebooks in a batch** (float-doc idea 2, `ivf_pq_build.cuh:349-416`). The pool is 68 s.
   **~34 s (16%) on its own, ~13 s after idea 1.** This is the main lever for the tiny corner-case builds (21.7 s).
   Keeps coverage. Effort high; risk medium, because recall may shift.
4. **Test: AnnCagraMultiPartitionTest builds each of its 13 partition sets once** (`buildPartitions`, `.cuh:2153-2173`).
   **~16 s (7.5%)**. Alternatively, fold `FilteredSearch` into `Search` (`.cu:34-35`): ~9 s. Keeps coverage. Low effort.
5. **Library: make NN-descent's per-iteration host update cheap for small graphs.** Use a persistent worker instead of a
   new `std::thread` per iteration (`nn_descent.cuh:2754, 2900`). Limit OpenMP threads by `nrow` in
   `update_graph` / `sample_graph` (:2232, :2277); for 500-10k rows the actual work should be sub-millisecond.
   **Up to ~15 s (7%)**, ~8 s after ideas 1 and 4. Keeps coverage. Low-medium effort, low risk. This idea is new here:
   the float doc only noted the thread.
6. **Test: FilterTest uses dim 16 instead of 8** (`.cuh:2017`, `:2036`). This moves 33 int8/uint8 cases to the batched
   path. Float is unaffected, since 8 and 16 are both aligned there. **~13 s (6%)**. Coverage barely changes: dims 1,
   17 and 102 still exercise the unaligned path. Idea 2 makes this unnecessary.
7. **Library: run the reverse-graph step for host graphs on the device** (float-doc idea 5, `graph_core.cuh:837-852`).
   **~8 s (4%)**, ~4 s after idea 1. Low effort, low risk.
8. **Library: skip zero-filling the kvikio staging buffer** (float-doc idea 6, `cpp/src/util/file_io.cpp:403`).
   **~4.3 s (2%)**, ~3 s after 1a. Low effort, low risk.
9. **Coverage-reducing (only if needed):** (a) the FilterTest MULTI_KERNEL `n_queries=100` cases (16 cases, 26.1 s)
   could use 10 queries: ~22 s. (b) The corner-case block could keep IVF_PQ only for SINGLE_CTA (11.7 + 15.3 s today):
   ~18 s, and 1c gets most of this without the loss. (c) A lower NN-descent `max_iterations` for tiny datasets: a few s.

Combined: 1c + 2 + 4 save ~120 s (58%). Adding 3, 5, 7 and 8 (after overlap) brings it to ~145 s (70%). The test-only
subset (1c, 4, 6) saves ~90 s (43%). Keep the executable in one process: JIT (15.5 s) and `CUFileInit` are paid per process.

## Uncertainties

* Phases are inferred, because CPU sampling was unavailable. Host time is charged to the phase of the preceding kernel
  launch, with ±10% at bucket edges. The serialize window is anchored on OSRT `pwrite` / `pread`.
* nsys/CUPTI inflate the cost per launch and per sync: there are 7.3 M launches, and FilterTest alone has 0.84 M blocking
  copies. Absolute seconds are therefore higher than in a plain run. The ranking should hold.
* The fallback savings (ideas 2 and 6) use float's batched dim-8 cases and int8's aligned-dim cases as the "after" cost.
  Build-reuse savings use the median build cost per key. Nothing was prototyped (no builds or GPU runs were allowed).
* The NN-descent figure (idea 5) rests on 20.6 s of measured `pthread_join` wait. How that splits between thread start,
  OpenMP fork/join and CPU contention from concurrent jobs is unknown. Why int8 InnerProduct/Cosine PHYSICAL merges wait
  ~5 ms per iteration (against 1.8 ms on average, and 1.5-6x slower than the same cases in float/uint8) is unexplained;
  tied int8 distances are one guess.
* JIT (15.5 s) and first-case start-up (~1.6 s per AnnCagraTest shard) are inflated by one process per group, and are not
  counted in any savings.
