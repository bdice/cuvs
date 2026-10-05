# NEIGHBORS_ANN_IVF_PQ_TEST: where the time goes and how to make it faster

Source: 17 nsys runs, one per fixture group (`-t cuda,nvtx,osrt`, RTX 6000 Ada). All times below come from
gtest under nsys. CUDA tracing inflates launch-bound code. The same executable took **496 s** under
`ctest -j8`, so seconds marked "≈ real" are computed as % × 496 s.

## Summary

| metric | value |
|---|---|
| cases / groups | 1397 / 17 (all fixtures in `ann_ivf_pq/test_{float,int8_t,uint8_t}_int64_t.cu`) |
| total gtest time (nsys) | 811 s (f32: 311 s, i08: 332 s, u08: 168 s) |
| GPU busy (kernels + copies + memsets) | 176 s = **22 %** of the CUDA span. The test is host/launch-bound. |
| kernel launches | **47.2 M** (≈ 34 k per case, mean kernel 4 µs) |
| biggest groups | i08 `build_serialize_search` 72.9 s, i08 `build_host_input_search` 70.0 s, i08 `build_host_input_overlap_search` 68.7 s, i08 `build_search` 64.0 s, f32 `build_serialize_search` 58.1 s |
| most expensive cases | `dim = 6144, pq_dim = 3072`: 8.4–12.2 s each, 11 cases = 110 s (13.5 %) |

## Where the time goes

I split each case into phases using kernel markers. Each case starts at its `naive_distance_kernel`. PQ training runs
from `copy_selected_kernel` to the last k-means kernel. The checks phase starts at
`reconstruct_list_data_kernel`/`unpack_list_data_kernel`, search starts at `compute_similarity`, and the rest is host time.

| phase (all 1397 cases) | s (nsys) | % | what it is |
|---|---|---|---|
| **PQ codebook training** | **632.6** | **78 %** | library `ivf_pq::detail::train_per_subset` (593 s) / `train_per_cluster` (40 s) |
| per-label checks in `run()` | 93.0 | 11.5 % | test `ann_ivf_pq.cuh:601-618` (reconstruct/erase/extend/pack per list) |
| coarse k-means + rotation | 32.9 | 4.1 % | library `kmeans::fit` with 32 lists |
| host recall evaluation | 27.7 | 3.4 % | test helper `calc_recall` in `ann_utils.cuh:228-280`; 21.3 s of it is k ≥ 1023 |
| extend / serialize / filter setup | 17.4 | 2.1 % | serialize ≈ 3.6–4 s per serialize group (kvikio, `CUFileInit` ≈ 1 s) |
| process start / first case, SetUp (`gen_data` + `naive_knn`), search | ≈ 17 | 2 % | `naive_distance_kernel` is only 1.3 s of GPU time in total; `compute_similarity` is 4 s |

* **The library's k-means is the whole story.** `train_per_subset` (`ivf_pq_build.cuh:349`) runs a full
  balanced k-means for each of the `pq_dim` subspaces. Each run does about 22–26 EM iterations
  (`kmeans_n_iters = 20` plus balancing pull-backs). The test's default `dim = 64` gives `pq_dim = 64` (pq_len 1).
  The big-dims cases have `pq_dim` up to 3072. The executable runs **4.24 M EM iterations** in total.
* **One EM iteration (measured on the 6144 case) costs 147 µs wall time but only 31 µs of GPU time.** Per iteration:
  10.5 kernel launches (≈ 42 µs of API time), 5.8 `cudaMallocFromPoolAsync` + 5.8 `cudaFreeAsync` (11 µs),
  1.4 memsets, and one *blocking* D2H copy plus sync. The copy is `adjust_centers`
  (`kmeans_balanced.cuh:811-812`): it copies the cluster sizes to the host and sorts them every iteration, even
  with the Random donor policy. About 66 µs per iteration is host time between calls. The rest of each
  iteration: `predict` recomputes the dataset norms every iteration (`kmeans_balanced.cuh:542`; `train_per_subset`
  passes no `dataset_norm`), and the unused `receiver/donor_clusters` buffers are still allocated (line 846).
* **Big dims cost 335 s (41 %) for only 110 cases** (8 %), and 297 s of that is PQ training (cost ∝ `pq_dim`).
  The 6144 case alone is 110 s. Near-duplicate dims (513, 1023, 1025, 2049, 2050) add 122 s.
* **Redundant builds.** Every case builds and checks its own index, but many cases only change *search*
  parameters: `.k`, `n_probes`, `lut_dtype`, `internal_distance_dtype`, `coarse_search_dtype`. Within each
  TEST_P, 868 of 1397 cases rebuild an index whose index-relevant parameters already appeared earlier in the same
  group: all of `var_k()` (17 per i08 group), 8 search-only entries per `enum_variety_*` block, and the default-equivalent
  entries. 332 cases are **exact duplicates** (ignoring `min_recall`): in `enum_variety()`, `{PER_SUBSPACE}`
  (`ann_ivf_pq.cuh:989`), `{pq_bits=8}` (1012), `{force_random_rotation=false}` (1021), `{lut_dtype=32F}` (1026) and
  `{internal_distance_dtype=32F}` (1050) all equal the default config (5 identical entries per metric block); u08
  instantiates both `enum_variety()` and `enum_variety_l2()` (`test_uint8_t_int64_t.cu:18,23`), identical because
  the default metric is L2Expanded; and `defaults()` repeats the L2 block's default entry.
* **Checks phase (93 s).** GPU: `encode_list_data_interleaved_kernel` (via `codepacker::extend_list`) takes 26 s,
  ~127 ms per launch for `pq_dim = 3072` with ~128 rows, i.e. poorly parallelised (`ivf_pq_build.cuh:682`).
  Host: `compare_vectors_l2` (`ann_ivf_pq.cuh:100-125`) allocates managed memory, then reads `dist(i)` element by
  element from the host, each read an 8-byte D2H `cudaMemcpyAsync` + sync (~170 k per group, ~1.9 k per case,
  ≈ 1.6 s of API time per group). Library: `recompute_internal_state` (`ivf_common.cuh:267-273`) issues 2·n_lists
  single-pointer H2D copies after every erase/extend (≈ 220 k per group, ≈ 0.5 s).
* **Host recall evaluation.** `calc_recall` is O(n_queries·k²) and is run twice: once for the distance-aware match
  and once for the index match. For k = 2048/2049 that is a 1.9 s CPU gap per case (seen as the
  `postprocess_neighbors_kernel → rngKernel` idle gaps), and 0.63 s for k = 1023. This affects the 5 i08 groups.
* **Not hotspots:** `naive_knn` (1.3 s), data generation, search kernels (4 s), process start (≈ 0.6 s). Lazy
  module loading is 2–2.7 s per group run (≤ 67 ms per `cuLibraryLoadData`), paid once in a single process.

## Ideas (ranked by standalone estimated savings; percentages are of this executable)

Savings overlap: B includes F, and A/C/E all shrink the same PQ-training time. Recommended combination that
keeps coverage: **B + H + I (+ E)** ≈ 40–50 % on the test side, then A for a large further win.

**A. Batch the per-subspace k-means in the library** *(keeps coverage)*
* Change: `ivf_pq_build.cuh:349-415` (`train_per_subset`) runs `pq_dim` independent `build_clusters` calls of
  identical shape (`pq_n_rows × pq_len`, `pq_book_size` centers). Train them together with one batched EM loop:
  one predict/update launch per iteration for all subspaces. Do the same for the `n_lists` loop in `train_per_cluster`
  (`:458`).
* Savings: up to ~60–70 % (≈ 300–350 s); PER_SUBSPACE training alone is 593 s (73 %). Effort: L. Risk: M
  (codebooks change numerically; IVF-PQ users with large `pq_dim` get faster builds too).

**B. Cache the built and checked index across cases that differ only in search parameters** *(keeps coverage)*
* Change: in `ivf_pq_test::run()` (`ann_ivf_pq.cuh:594`), wrap `build_index()` + the per-label checks (601-618) in a
  small static cache. Key: fixture type, TEST_P name, `num_db_vecs`, `dim`, and all `index_params`. Then run
  `search` on the cached `const` index. Searches don't mutate it, and the cached state equals what the case would
  have built and checked itself. Do the same for `ivf_pq_filter_test::run()` (795) and `build_precomputed()` (270,
  which has no search at all).
* Every search-parameter variant still searches and evaluates recall, and every unique index is still built,
  serialized, extended and checked once.
* Savings: 298 s unbounded (**37 %, ≈ 180 s real**), 263 s with LRU-4, 235 s keeping only the last index.
  Big-dim keys are unique, so skip caching for dim ≥ 512 to bound memory (~170 MB per 6144-dim index).
* Effort: M (~60 lines). Risk: L–M (cross-case state; gtest filtering or shuffling only lowers the hit rate).

**C. Use fewer k-means iterations in the tests** *(keeps code paths, changes the configuration)*
* Change: set `index_params.kmeans_n_iters` in the `ivf_pq_inputs` constructor (`ann_ivf_pq.cuh:39-42`), e.g.
  20 → 10, or only in `big_dims()` / `big_dims_moderate_lut()` (922, 937), e.g. → 5. EM iterations scale with this
  value, and balancing still runs from iteration 1.
* Savings: ≈ 35–40 % with 10 everywhere; ≈ 25 % (≈ 125 s) with 5 for big dims only. Effort: S. Risk: **M**:
  the `min_recall` thresholds (0.86 for enum, big-dims formula at `:930`) must be revalidated on the GPU, and
  centers start from mod-labels so they need iterations to spread out.

**D. Run big dims in fewer build paths** *(reduces coverage)*
* Change: `big_dims*()` is crossed with all 5–6 build paths per type (`test_float_int64_t.cu:19-21`,
  `test_int8_t_int64_t.cu:19-21`); keep it in one device-build test + the filter test per type.
* Savings: 218 s (27 %, ≈ 133 s). Effort: S. Risk: L. Loses host-input/serialize/extend × large-dim combinations.

**E. Cut per-iteration overhead in the library's balanced k-means** *(keeps coverage)*
* Change, in `kmeans_balanced.cuh`:
  * (a) In the Random donor path of `adjust_centers`, skip the D2H copy + sync + host sort (811-819) and the
    unused `receiver/donor` allocations (846). The kernel already filters on `lower_threshold`.
  * (b) Compute the dataset norms once per `build_clusters` and pass them as `dataset_norm` (`predict` at 542).
  * (c) Reuse the EM workspaces instead of ~6 alloc/free pairs per iteration.
* Savings: ≈ 10–20 % (≈ 50–100 s). Effort: M. Risk: L–M ((a) changes how the static `i_primes` (805) advances,
  so results differ but remain valid).

**F. Delete the exact-duplicate parameter sets** *(keeps coverage; subsumed by B)*
* Change: drop the five default-equivalent entries listed above from `enum_variety()` (keep one, with the
  strictest `min_recall`). Drop `enum_variety()` from the u08 instantiations, and `defaults()` where an L2 block
  follows.
* Savings: 332 cases, 121 s (**15 %**, ≈ 74 s). Effort: S. Risk: L.

**G. Drop `dim = 6144` from `big_dims()`** (`ann_ivf_pq.cuh:925`) *(reduces coverage)*
* Savings: 110 s (13.5 %, ≈ 67 s); dropping near-duplicate dims 513/1023/1025/2049/2050 would add 122 s (15 %).
  The LUT is already beyond shared memory at 2048 dims, so the search kernel path is probably the same
  (unverified). Effort: S. Risk: L. Overlaps D.

**H. Make the host recall check O(k log k)** *(keeps coverage)*
* Change: in `calc_recall` (`ann_utils.cuh:228-280`), use a per-row hash set for index matches, and sorted
  expected distances + binary search within eps for distance matches, instead of the k² double loop run twice.
* Savings: 21–28 s (≈ 3 %, ≈ 15 s); shared helper, so other ANN tests benefit too. Effort: S. Risk: L.

**I. Fix `compare_vectors_l2`** (`ann_ivf_pq.cuh:100-125`) *(keeps coverage)*
* Change: compute the max error on the device (or copy `dist` once to a `std::vector`) instead of using a managed
  array with per-element host reads (one D2H copy + sync each, ~1.9 k per case).
* Savings: ≈ 20 s (≈ 2.5 %); about half that if B is applied. Effort: S. Risk: L.

**J. Big-dim checks: the slow encode kernel** *(library fix keeps coverage)*
* Change: `encode_list_data_interleaved_kernel` (`ivf_pq_build.cuh:682`, via `codepacker::extend_list`) takes
  ~127 ms per launch for 128 rows × 3072 subspaces. Parallelise it over subspaces.
* Savings: 26 s GPU (3.2 %). Effort: M. Risk: L. Test-only alternative *(reduces coverage)*: run
  `check_reconstruct_extend` on 2–3 labels when dim ≥ 512.

**K. Minor fixes** *(keep coverage; each < 1 %)*
* Batch the pointer copies in `recompute_internal_state` (`ivf_common.cuh:267-273`): ≈ 6 s.
* Serialization via kvikio temp files (`build_serialize`, `ann_ivf_pq.cuh:261`; `CUFileInit` ≈ 1 s + per-file
  handle registration): ≈ 4 s per serialize group.

## Uncertainties

* **nsys inflation.** All per-phase times were measured under nsys CUDA tracing. Groups summed to 811 s of gtest
  time, versus 496 s for the whole executable under contended `ctest -j8`. Launch-bound phases (PQ training) are
  probably inflated more than GPU-bound ones, so the real share of PQ training may be somewhat below 78 %.
  Converting with % × 496 s is an approximation. nsys wall time per group (69–167 s) also includes nsys
  finalisation, which real runs don't have.
* **Phase boundaries are inferred**, not measured. The export has no cuvs-domain NVTX ranges (only cub/thrust/kvikio),
  so boundaries come from kernel-name markers and are good to a few ms per case. For extend/precomputed tests the
  "PQ training" window can include part of `extend`.
* **Host-gap attribution comes from reading code.** CPU sampling was unavailable, so the 1.9 s gaps are attributed
  to `calc_recall`'s O(k²) loops, and the per-element D2H copies to `compare_vectors_l2`'s managed `dist(i)` reads.
* **Cache savings (B, F)** are computed from measured per-case build + check time of repeated index keys. They
  assume a cache hit costs only SetUp + search + evaluation. Builds were already not bit-reproducible across cases
  because of the process-wide `static i_primes` in `adjust_centers`.
* **Not validated on the GPU** (none could be run here): the recall impact of fewer k-means iterations (C), the
  library estimates (A, E; design estimates), and whether `dim = 6144` exercises a unique search path (G).
