# NEIGHBORS_ANN_CAGRA_HALF_UINT32_TEST: where the time goes and how to make it faster

Source: 19 `nsys profile -t cuda,nvtx,osrt` runs, one fixture group (or gtest shard) per process, on an RTX 6000 Ada,
branch `staging-test-optimizations-local` (#2685, #2726). gtest times under nsys sum to 257 s; under `ctest -j8`, with
other tests sharing the GPU, the executable took 200 s. `.cu` = `cpp/tests/neighbors/ann_cagra/test_half_uint32_t.cu`,
`.cuh` = `cpp/tests/neighbors/ann_cagra.cuh`. The shared findings are explained in the
[float doc](NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST.md) and only quantified here. How half differs:
* **Fixtures.** Half has the same `_U32`/`_I64` doubling as float (`.cu:13-14, 20-21`), but no FilterTest, no
  FilteredMerge and no CagraQ or MP-rejection tests. The I64 twins cost 118 s (46%) here. The uint8/int8 executables
  have no I64 twins, but they do run FilterTest (36 s each).
* **Dims ≥ 256 never run.** `fpi_mapper<half>::kBitshiftBase` is 11 (`.cuh:172-176`), so
  `GenerateRoundingErrorFreeDataset` calls `GTEST_SKIP` in SetUp for every dim ≥ 256 (`.cuh:210-212`). That covers 218
  of the 1986 cases: 48 per AnnCagraTest/IndexMerge TEST_P, 24 in AddNodes and 1 per MP TEST_P. Float runs 132 of these
  and spends 38 s on them. This is a coverage gap, not a cost. Dims 128-255 get only 4 distinct values per coordinate.
* **The alignment rule is `dim % 8 == 0`.** The batched search needs a 16 B row pitch (`common.hpp:1041-1049`,
  `cagra_search.cuh:114`). The unaligned dims are the same as for float (1, 3, 7, 17, 137); dim 8 is aligned, unlike
  for uint8/int8.
* **The cost per case does not depend on dtype.** The 1114 cases that run in both executables take 256.7 s in half and
  252.5 s in float (+1.7%). Half-typed kernels (`naive_distance_kernel<float, half>`, the EFT generator,
  `fused_1nn_h_i32`, f16 GEMMs) total < 0.1 s of GPU time, and there are no host conversions. Only IVF-PQ at dims 64
  and 192 is 1.3-1.9x slower than in float, which adds ~3.5 s.

## Summary

| TEST_P (`.cu` line) | cases (skipped; of which dim≥256) | gtest s | % | GPU busy | kernel launches | dominant cost |
|---|---|---|---|---|---|---|
| AnnCagraIndexMerge_I64 (:21) | 459 (225; 48) | 70.3 | 27 | 17% | 1.93 M | IVF-PQ 37.6 s. It repeats the `_U32` builds |
| AnnCagraIndexMerge_U32 (:20) | 459 (225; 48) | 65.1 | 25 | 18% | 1.92 M | IVF-PQ 35.4 s, 2-3 builds per case |
| AnnCagra_U32 (:13) | 459 (189; 48) | 47.4 | 18 | 13% | 0.85 M | IVF-PQ 18.2 s, serialize 7.3 s |
| AnnCagra_I64 (:14) | 459 (189; 48) | 47.4 | 18 | 13% | 0.86 M | the same builds as `_U32` |
| MultiPartition Search + FilteredSearch (:32-33) | 24 + 24 (1 + 1) | 10.2 + 8.1 | 7 | 15% | 0.05 M | NN-descent: 244 partition builds for 12 sets |
| AnnCagraAddNodes (:17) | 102 (42; 24) | 8.5 | 3 | 78%* | 0.10 M | iterative build, one query at a time: 5.7 s |
| **Total** | **1986 (872; 0.2 s)** | **257** | | **18%** (45.5 of 259 s) | **5.71 M** | host / launch bound |

\*Misleading: 72.9k `search_single_cta` launches have a 1-block grid (76 µs each, 5.6 s), whereas a batched 900-query
launch takes 92 µs. Largest shards: Merge_I64 4/4 (20.2 s), 1/4 (17.7 s), 3/4 (17.5 s).

## Where the time goes

Method: as in the float doc. Each executed case was cut out at its two `GenerateRoundingErrorFreeDataset_kernel`
launches; the windows sum to 256.8 s against 257.0 s of gtest time. Main-thread time was charged to the family of the
last distinctive kernel. JIT (the host gap before `cu/cudaLibraryLoadData`), the per-case kvikio span (first to last
kvikio NVTX range) and `CUFileInit` were carved out. Searches after a case's last `optimize` kernel are the test's own.

| bucket | s | % | float doc | detail |
|---|---|---|---|---|
| IVF-PQ graph build | 109.0 | 42.5 | 43% | ~100-120 ms and ~3.7k launches per build at 1000x1 and 1000x16. 264 ms and 16.9k launches at 500x192. ~300 ms per merge case. |
| NN-descent | 32.3 | 12.6 | 13% | 758 builds at ~43 ms. MultiPartition: 11.3 s (244 partition builds of 10k rows). |
| CAGRA search, the test's own | 22.7 | 8.8 | 12% | **19.7 s from `dim % 8 != 0`.** Unaligned MULTI_KERNEL: 9.7 s, 184 ms per merge case against 66 ms aligned. |
| search inside the iterative build | 16.9 | 6.6 | 5% | **16.8 s from unaligned dims**: 137 ms per iterative-build case against 1.3 ms aligned. AddNodes: 5.7 s. |
| JIT link (nvJitLink) | 25.3 | 9.8 | 9% | ~100 loads, ~1.5 s per shard. Mostly an artifact of one process per group. |
| CAGRA graph optimize | 18.8 | 7.3 | 7% | 11.0 s is the host-graph reverse-graph loop: 2616 runs x 4.2 ms, with a sync per column. |
| serialize + deserialize | 14.5 | 5.6 | 7%† | 20.9 ms per case (median). 576 host-only gaps after the kvikio `FileHandle` open (median 15.6 ms) total 9.4 s: the 32 MiB zero-fill. |
| `CUFileInit` | 8.0 | 3.1 | † | ~1 s once per AnnCagraTest process (8 processes). An artifact. |
| reference + verification + data generation | 6.2 | 2.4 | 1.2% | naive_knn, eval_distances, InitDataset, SetUp host time. **Not worth optimizing.** |
| IVF-PQ search during the build, merge scaffold | 3.2 | 1.2 | 2% | |

†The float doc's 24.0 s for serialize includes `CUFileInit`. Measured the same way, half's is 22.5 s.

## Ideas (ranked by savings after idea 1; % of 257 s; savings overlap)

1. **Build each index once and reuse it. Keeps coverage, test-only** (float-doc idea 1, same code locations):
   a. Fold `AnnCagraIndexMerge_I64` into `_U32` (`.cu:20-21`): build and merge once, then search into uint32 and int64
      outputs. **55 s (21%)**. Low effort, low risk.
   b. Fold `AnnCagra_I64` into `_U32` (`.cu:13-14`), which also serializes once. **33.5 s (13%)**. Low effort, low risk.
   c. Drop the `merge_strategy`/`host_dataset` axes from AnnCagraTest (own input list or a filter at `.cu:23`). That
      removes 94 duplicate executed cases per TEST_P. **12 s (5%) after 1b**, 25 s on its own. No risk.
   d. Reuse the two half-indices between the 70 PHYSICAL/LOGICAL twin pairs in IndexMerge. **12 s (5%) after 1a**.
   a-d: **~114 s (44%)**. Option e, a process-wide index cache (float-doc 1e), saves **~131 s (51%)**. Today 270
   AnnCagraTest cases use 93 distinct builds, and 234 IndexMerge cases use 87 build keys. The cache also covers the
   corner-case block. Medium effort. Risk: a failing build fails every case that shares it.
2. **Library: batch the queries when the padded query stride != dim** (float-doc idea 3; `cagra_search.cuh:114,
   141-160`; `jit_lto_kernels/setup_workspace_impl.cuh:53, 143`). For half this triggers at `dim % 8 != 0`. The pool is
   36.5 s; at aligned per-case cost ~6 s would remain. **~30 s (12%)**, still ~23 s after idea 1 because the folded I64
   searches stay. Keeps coverage. Effort and risk medium. *Partial fix:* poll `terminate_flag` every N iterations
   (`search_multi_kernel.cuh:586-592`): ~5 s of the 9.7 s unaligned MULTI_KERNEL pool.
3. **Library: train the IVF-PQ codebooks in a batch** (float-doc idea 2, `ivf_pq_build.cuh:349-416`). The pool is 109 s.
   If half of it is launch/sync overhead, **~55 s (21%)** on its own. After idea 1 only 18-24 s of IVF-PQ build remains,
   so ~12 s. Keeps coverage. It ranks below 2 because it is unmeasured, high effort, and recall may shift.
4. **MultiPartition: build each partition set once** (`buildPartitions`, `.cuh:2153-2173`; 46 cases use 12 sets).
   **~11 s (4.3%)**. Folding `FilteredSearch` into `Search` (`.cu:32-33`) alone saves 7.5 s. Keeps coverage. Low effort, low risk.
5. **Library: build the reverse graph on the device for host-resident graphs** (`graph_core.cuh:836-852`; reuse the
   device path at :828-834). **~10 s (3.9%)**, ~5 s after idea 1. Low effort, low risk.
6. **Library: stop zero-filling the 32 MiB kvikio_ofstream buffer** (`cpp/src/util/file_io.cpp:265, 403`;
   `include/cuvs/util/file_io.hpp:466`). **9.4 s (3.6%)**, ~4.7 s after 1b. Low effort, low risk.
7. **Reduces coverage:** in the corner-case block (`.cuh:1806-1826`), keep IVF_PQ only for SINGLE_CTA. **~35 s (14%)**,
   or ~17 s after 1a/1b. Idea 1e gets this without losing coverage.
8. **Half-specific, no time saved:** the 218 dim ≥ 256 cases give no half coverage. Filter them out of the half lists so
   the gap is explicit. A half generator that works at high dims would add ~38 s (~20 s after idea 1).

Combined: ideas 1e, 2, 4, 5 and 6 save ~175 s (68%), or ~185 s (72%) with idea 3. Keep the executable in one process:
the 25 s of JIT and 8 s of `CUFileInit` come from 19 processes (single process: est. 3-5 s, not counted above).

## Uncertainties

* Phases are inferred, because CPU sampling was unavailable. Bucket edges are approximate (about ±10%). nsys/CUPTI
  inflate per-launch and per-sync cost (5.7 M launches), so absolute seconds are high; percentages and the ranking should
  transfer.
* The float comparison uses exact per-case gtest times matched by case index; the `value_param` strings are identical.
  Float phases were not re-measured, because their sqlite exports were no longer available.
* The ~3.5 s IVF-PQ slowdown at dims 64/192 is unexplained. One hypothesis: the low-entropy EFT data (4-8 values per
  coordinate) gives balanced k-means more work. It is unverified, because float launch counts were unavailable.
* The library savings (ideas 2, 3, 5, 6) assume that the fixed code runs at the cost of today's aligned / device-path
  equivalents. Nothing was prototyped (no builds or GPU runs). Build reuse assumes a shared index is acceptable coverage.
