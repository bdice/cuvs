# NEIGHBORS_ANN_CAGRA_UINT8_UINT32_TEST: where the time goes and how to make it faster

Source: 12 `nsys profile -t cuda,nvtx,osrt` runs, one fixture group (or gtest shard) per process, on an RTX 6000 Ada,
branch `staging-test-optimizations-local` (includes #2685 and #2726). All times are gtest times measured under nsys.
The sum is 208 s; for comparison, the whole executable took 265 s under `ctest -j8` with other tests sharing the GPU.
Tests: `cpp/tests/neighbors/ann_cagra/test_uint8_t_uint32_t.cu` (`.cu`). Fixtures: `cpp/tests/neighbors/ann_cagra.cuh` (`.cuh`).
This executable runs the same fixtures as the float one, so most findings are shared. They are explained in
[NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST.md](NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST.md) ("float doc") and only quantified
for uint8 here. The uint8 executable differs in four ways:
* It has one TEST_P per fixture. There is no `_U32`/`_I64` doubling, no FilteredMerge and no CagraQ, so float-doc
  ideas 1a/1b do not apply.
* **The CAGRA one-query-at-a-time fallback hits more dims.** A batch needs the row pitch `dim * sizeof(T)` to be a multiple
  of 16 B. For uint8 that means `dim % 16 == 0` (for float, `dim % 4 == 0`), so dims 8 and 10 also take the slow path.
* BitwiseHamming actually runs here; other types skip it.
* The time distribution differs: FilterTest costs 36 s instead of 22.7 s.

## Summary

| TEST_P (`.cu` line) | cases (skipped) | gtest s | % | GPU busy | kernel launches | dominant cost |
|---|---|---|---|---|---|---|
| AnnCagraIndexMergeTest (:28) | 459 (174) | 82.8 | 40 | 18% | 2.61 M | IVF-PQ graph builds: 59.4 s in 134 cases |
| AnnCagraTest (:21) | 459 (138) | 56.2 | 27 | 14% | 1.16 M | IVF-PQ builds 38.4 s; 179 cases are indistinguishable duplicates |
| AnnCagraFilterTest (:25) | 60 (0) | 36.0 | 17 | 42% | 3.36 M | MULTI_KERNEL one query at a time: 26.7 s in 20 cases |
| AnnCagraMultiPartitionTest Search + FilteredSearch (:34-35) | 24 + 24 | 10.7 + 9.9 | 10 | 15% | 0.05 M | NN-descent on 10k rows, partitions rebuilt per case |
| AnnCagraAddNodesTest (:22) | 102 (9) | 12.1 | 6 | 82%* | 0.15 M | iterative build, one query at a time: 10.2 s |
| **Total** | **1128 (321 skipped, 0.2 s)** | **208** | | **24%** (51 s) | **7.3 M** | host / launch bound |

\*GPU busy is misleading here: 105k `search_single_cta` launches with a 1-block grid (77 µs each, 8.1 s) run back to back.

## Where the time goes

Method: each case was cut out of the timeline using cumulative gtest durations, anchored at the first CUDA call. The cases
were then mapped to their `AnnCagraInputs`, which were regenerated from the `.cuh` blocks. JIT link time is the span from
`cuInit` to `cudaLibraryLoadData`. The serialize gap is the host-only gap after the kvikio `open`.

| bucket | s | % | detail |
|---|---|---|---|
| IVF-PQ-built cases (build, search, serialize, merge) | 97.8 | 47 | AnnCagraTest takes 0.2 s per case and IndexMerge 0.4 s (two half-builds plus merge). That is ~18k launches and 1.6k syncs per merge case, with GPU busy at 8-13%. Dim 1024 costs 1.9 s per merge case. For the cause (serial PQ codebook k-means), see the float doc. |
| **One-query-at-a-time fallback** (uint8 dims 1, 3, 7, 8, 10, 17, 102, 137) | **~51** (excess) | 25 | `cagra_search.cuh:114` sets `can_batch_n_queries` to false, then runs one plan per query (:141-159). 205k of the 206k `search_single_cta` launches have a 1-block grid (20.9 s of GPU time). Breakdown: FilterTest ~30 s, AddNodes ~9 s (iterative build: 900-1000 single-query searches per pass), AnnCagraTest + IndexMerge ~12 s. |
| ↳ FilterTest MULTI_KERNEL | 26.7 | 13 | 20 cases. Each search runs 100 queries x ~256 iterations x (4 kernels + D2H `get_value` + sync), at `search_multi_kernel.cuh:543-617`. That adds up to 827k iterations and 3.3 M launches. Each iteration costs ~31 µs of host time: 15 µs blocking memcpy plus launches. The single-query `search_single_cta` takes 1.07 ms per query (4.3 s). |
| ↳ uint8-only part (dim 8) | +15.2 | 7 | The same dim-8 cases cost 18.7 s in uint8 but 3.5 s in float, where dim 8 is aligned. That is 33 FilterTest cases (16.6 vs 3.5 s) and 12 AddNodes cases (2.1 vs 0.1 s). |
| cases AnnCagraTest cannot distinguish | 18.4 | 8.8 | 179 cases (118 executed). They differ only in `merge_strategy`, `host_dataset` or `itopk_size`, which AnnCagraTest ignores (`.cuh:451-455`, `:467-476`). IndexMerge ignores `host_dataset`: 6 cases, 2.3 s. |
| the same index rebuilt for another search/merge variant | ~45 | 22 | AnnCagraTest: 203 unique executed cases for 117 build keys (~10 s). IndexMerge: 285 executed cases for 111 keys (~34 s). PHYSICAL/LOGICAL twins alone are 94 cases, 22.0 s. FilterTest: 60 cases for 13 builds (~3 s). |
| MultiPartition NN-descent builds | ~19 | 9 | 48 builds for 13 distinct partition sets. GPU busy is only 0.05-0.08 s per case; the rest is host time (float doc: one `std::thread` per NN-descent iteration). |
| JIT link (nvJitLink) | 14.4 | 7 | ~55 links per shard at ~27 ms each, across 12 processes. One ctest process pays a large share of this only once. |
| reverse-graph loop in `optimize` | ~8 | 4 | 79.5k per-column iterations: OpenMP gather, H2D copy, kernel, sync (`graph_core.cuh:836-852`). |
| serialize staging-buffer zero-fill | 5.5 | 2.6 | 17 ms per executed AnnCagraTest case (321 cases). This is the float doc's 32 MiB `std::vector<char>`. |
| first case per process (context, CUFileInit, first JIT) | ~10 | 5 | 1.6 s in AnnCagraTest shards and 0.9 s in IndexMerge shards, against a median case of 0.1 s. It overlaps with JIT. |
| BitwiseHamming (uint8-only coverage) | 6.9 | 3.3 | 117 cases. The 56 skipped by `k*dim*8/5 < n_rows` cost ~0; the 61 that run are not hot. **Keep.** |
| test reference, verification and data generation | <1 | <0.5 | `naive_knn`, `eval_distances`, `eval_neighbours`, uniformInt/IP normalization. **Not worth optimizing.** |

## Ideas (ranked by estimated savings; % of the 208 s; the ideas overlap, so a combined figure is given at the end)

1. **Build each index once and reuse it (test-only). Keeps coverage.** The design and its caveats are in float-doc idea 1.
   a. *Drop the AnnCagraTest cases that differ only in `merge_strategy` / `host_dataset` / `itopk_size`.* Filter `inputs`
      at INSTANTIATE (`.cu:21`) by an effective key, or give AnnCagraTest its own list. The axes are at `.cuh:1824`,
      `:1852`, `:1881`, `:1930`, `:1949`. **18.4 s (8.8%)**. Low effort, no risk.
   b. *IndexMerge: build the two halves once per case and run both PHYSICAL and LOGICAL on them* (`.cuh:1555-1601`).
      This collapses the 94 twin cases (22.0 s). **~18 s (8.7%)**. Low effort, low risk.
   c. *Process-wide index cache* keyed by (n_rows, dim, metric, degree, build_algo, refine, slice), used in AnnCagraTest,
      IndexMerge and FilterTest. **~60 s (29%) including a and b.** Medium effort. Risk: shared state, and a failing build
      fails every case that uses it.
2. **Library: batch queries when the padded query stride != dim. Keeps coverage, and is the largest uint8-specific item.**
   Pass a query leading dimension into `setup_workspace` (`jit_lto_kernels/setup_workspace_impl.cuh:53, 143`:
   `queries_ptr += dim * query_id`) and the multi-kernel / random_pickup kernels. Then drop the per-query loop at
   `cagra_search.cuh:95-160`. **~50 s (24%)**: FilterTest 36 → ~6 s, AddNodes 12 → ~3 s, and ~12 s in AnnCagraTest +
   IndexMerge. This also speeds up the iterative-build path for real users with uint8/int8 data and `dim % 16 != 0`.
   Effort: medium (JIT fragment signatures change). Risk: medium, because the per-query path was added to avoid
   misaligned query reads (the comment at `cagra_search.cuh:95-100`). The same fix also speeds up
   NEIGHBORS_ANN_CAGRA_INT8_UINT32_TEST, whose FilterTest also takes 36.1 s.
   *Cheaper partial fix:* read `terminate_flag` every N iterations instead of doing a blocking D2H plus sync on every one
   (`search_multi_kernel.cuh:586-592`). This removes ~15 of the ~31 µs per iteration: **~10-14 s (5-7%)**. Low effort,
   low risk.
3. **Library: train the IVF-PQ codebooks in a batch** (float-doc idea 2, `ivf_pq_build.cuh:349-416`). IVF-PQ-built cases
   are 47% of this executable. **~35 s (17%) on its own, ~15 s after idea 1.** Keeps coverage. Effort high; risk
   medium, because recall may shift.
4. **AnnCagraMultiPartitionTest: build each of the 13 partition sets once** (`buildPartitions`, `.cuh:2153-2173`), or
   fold `FilteredSearch` into `Search` (`.cu:34-35`, ~9 s on its own). **~13 s (6%)**. Keeps coverage. Effort low; risk low.
5. **FilterTest: use dim 16 instead of dim 8** (`.cuh:2017` and `:2036`). For uint8/int8 this moves 33 cases from the
   per-query path to the batched path. Float is unaffected (8 and 16 are both aligned there). **~13 s (6%)**. Coverage changes slightly:
   dim 8 is no longer tested in FilterTest, but dims 1, 17 and 102 still cover the unaligned path. Low effort, low risk.
   Idea 2 makes this unnecessary.
6. **Library: run the reverse-graph step for host graphs on the device** (float-doc idea 5, `graph_core.cuh:836-852`).
   **~6 s (3%)**, ~3 s after idea 1. Keeps coverage. Low effort, low risk.
7. **Library: skip zero-filling the kvikio staging buffer** (float-doc idea 6, `cpp/src/util/file_io.cpp:265, 403`).
   **5.5 s (2.6%)**, ~3.5 s after 1a. Keeps coverage. Low effort, low risk.
8. **Coverage-reducing options (only if 1-5 are not enough):**
   (a) The FilterTest MULTI_KERNEL rows with `n_queries=100` at unaligned dims are 16 cases costing 25.9 s. Use 10
   queries: ~22 s. Or use `itopk_size` 64 (`.cuh:2024`): ~19 s, but the 0.995 recall may not hold.
   (b) In the corner-case block (`.cuh:1806-1826`), the IVF_PQ cases cost 11.8 s in AnnCagraTest and 15.0 s in IndexMerge.
   Keep IVF_PQ only for SINGLE_CTA: ~18 s. Idea 1c gets most of this without losing coverage.

Combined: 1c + 2 + 4 save ~115 s (55%). Adding 3, 6 and 7 brings it to ~135 s (65%). The test-only subset (1c, 4, 5)
saves ~85 s (41%). Keep the executable in one process: in this profile, ~10 s of start-up and most of the 14.4 s of JIT
come from running 12 separate processes.

## Uncertainties

* Phases are inferred, because CPU sampling was unavailable and cuVS NVTX ranges are not compiled in. Case windows come
  from gtest durations at ms precision, with an error of ≤60 ms per shard. The IVF-PQ / NN-descent split inside a case
  is taken from the float doc and was not re-measured.
* The fallback savings assume that unaligned cases would then cost what the aligned twins cost. Those twins are the same
  cases in the float run (dim 8) or neighbouring aligned dims in the same block. Nothing was prototyped (no builds or GPU
  runs were allowed).
* nsys/CUPTI inflate the cost per launch and per sync. FilterTest alone has 3.4 M launches and 0.83 M syncs, so its
  absolute seconds, and the savings from idea 2, are likely lower without nsys. The ranking should still hold.
* Build-reuse savings (idea 1) use the cheapest case of each build key as the build cost minus 30-40 ms of
  search/verification. This may count some search time as build time when a key's cheapest case uses an unaligned dim.
* JIT link (14.4 s) and first-case start-up (~10 s) are inflated by running one process per group. Single-process values
  were not measured, and they are not counted in any savings above.
