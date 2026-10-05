# NEIGHBORS_ANN_* gtests: combined profiling summary

This combines the 19 per-executable analyses in this directory, which are the source of truth: every figure is
copied from them or derived by simple arithmetic. Setup: Nsight Systems, one fixture group per process, RTX 6000 Ada, branch
`staging-test-optimizations-local` (#2685, #2726, async RMM).

**Units.** `n` = gtest seconds under nsys. `c` = seconds at `ctest -j8` scale, either quoted by the analysis ("≈ real") or scaled here as
`n × min(1, ctest/nsys)`. Totals are in `c`. The ~2,470 s is a sum of per-test times in one contended `ctest -j8` run, not wall-clock.

## Executive summary

* **Where the time goes.** IVF_PQ, VAMANA, CAGRA ×4 and IVF_FLAT take 2,117 s (86%). Most of that is library index builds: k-means and PQ codebook
  training (including CAGRA's IVF-PQ graph build), NN-descent, and Vamana GreedySearch. The tests multiply it by rebuilding the same index for every
  search-only variant. Their own references (`naive_knn`, `eval_*`) take ≤ 2% everywhere.
* **Host- and launch-bound.** GPU busy is ~15–26% in 11 executables, 4–10% in RaBitQ and HNSW-ACE, and < 1% in FILTER_UDF. VAMANA shows 81% but runs ≤
  15 blocks per launch on 142 SMs. The common pattern is µs-scale kernels plus a blocking D2H + sync in every iteration.
* **Exceptions.** Test-side checks are a large share in IVF_RABITQ (70%, `calc_recall`), NN_DESCENT (26%), IVF_FLAT (20%, packer check) and IVF_PQ
  (11.5%). JIT dominates FILTER_UDF (72%) and TEST_BUGS (57%, cold cache).
* **Biggest levers** (standalone, overlapping): test-side index reuse and dedupe ≈ 745 c; k-means/PQ training ≈ 525–595 c (design estimate, 300–350 of
  it IVF-PQ); CAGRA unaligned-dim batching ≈ 170 c; serialization ≈ 130 c; host verification ≈ 125 c.
* **Combined, coverage-keeping.** Test-only changes save ≈ 885 c (~36%). Adding the library fixes brings it to ≈ 1,300–1,350 c (~55%). That excludes
  IVF-PQ batched training after reuse and Vamana occupancy, which were not quantified. **Nothing was validated on a GPU.**

## 1. Overview

Executable names link to the analyses. Cases = instantiated (skipped).

| executable | ctest s | nsys s | cases | GPU busy | dominant cost | best coverage-keeping idea (est.) |
|---|---|---|---|---|---|---|
| [IVF_PQ](NEIGHBORS_ANN_IVF_PQ_TEST.md) | 496 | 811 | 1,397 | 22% | library per-subspace PQ k-means 78% | index cache across search-only cases: 298 n ≈ 180 c; library batched PQ k-means ≈ 300–350 c |
| [VAMANA](NEIGHBORS_ANN_VAMANA_TEST.md) | 317 | 339 | 1,260 | 81%¹ | library `GreedySearchKernel` 66% | no kvikio buffer zero-fill ≈ 27 c |
| [CAGRA_FLOAT](NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST.md) | 313 | 366 | 2,520 (1,064) | 18% | IVF-PQ graph build 43%; U32/I64 twins | index cache + U32/I64 fold: ≈ 185 n ≈ 158 c |
| [IVF_FLAT](NEIGHBORS_ANN_IVF_FLAT_TEST.md) | 275 | 361 | 390 | 24–26% | k-means 3×/case 37%, per-list I/O 27%, packer 20% | train once per case: 88 n ≈ 67 c; recommended set ≈ 180 c |
| [CAGRA_UINT8](NEIGHBORS_ANN_CAGRA_UINT8_UINT32_TEST.md) | 265 | 208 | 1,128 (321) | 24% | IVF-PQ cases 47%; unaligned-dim fallback ~25% | index cache ≈ 60 n; unaligned batching ≈ 50 n |
| [CAGRA_INT8](NEIGHBORS_ANN_CAGRA_INT8_UINT32_TEST.md) | 251 | 208 | 1,128 (382) | 23% | IVF-PQ 33%, fallback 21%, NN-descent 17% | index cache ≈ 54–68 n; unaligned batching ≈ 51 n |
| [CAGRA_HALF](NEIGHBORS_ANN_CAGRA_HALF_UINT32_TEST.md) | 200 | 257 | 1,986 (872) | 18% | IVF-PQ 42.5%; I64 twins 46% | index cache + I64 fold: ≈ 131 n ≈ 102 c |
| [NN_DESCENT](NEIGHBORS_ANN_NN_DESCENT_TEST.md) | 76 | 103 | 916 (336)² | 19% | host loop 57%; verification 26% | linear helpers ≈ 24 n ≈ 18 c; thread/OpenMP fix 20–35 n |
| [IVF_RABITQ](NEIGHBORS_ANN_IVF_RABITQ_TEST.md) | 55 | 58 | 230 | 5–7% | `calc_recall`, one k = 16384 case ×5: 70% | O(k log k) `calc_recall` ≈ 39.5 (≈ real) |
| [CAGRA_FILTER_UDF](NEIGHBORS_ANN_CAGRA_FILTER_UDF_TEST.md) | 49³ | 16.9 | 24 | < 1% | nvJitLink re-LTO per UDF source 72% | persisted JIT cache ≈ 10.8; one parameterised UDF ≈ 6 (vs ~15 s single-process) |
| [IVF_SQ](NEIGHBORS_ANN_IVF_SQ_TEST.md) | 45 | 59 | 135 | 19% / 11% | k-means 2×/case 40%; (de)serialize 40% | reuse build 1's quantizer: 12.0 n ≈ 9 c; recommended set ≈ 26 c |
| [CAGRA_BUGS_TEST](NEIGHBORS_ANN_CAGRA_BUGS_TEST.md) | 36 | 18.9⁴ | 17 | — | cold JIT 57% of ctest; 1.18 M-row NN-descent | synthetic graph in the MultiCTA reproducer ≈ 9.5–10 c |
| [CAGRA_BBQ](NEIGHBORS_ANN_CAGRA_BBQ_UINT32_TEST.md) | 23 | 16.4 | 135 | ~15% | NN-descent host loop 64% (135 builds, 27 graphs) | build once per param + dense reference per metric ≈ 11.5 n |
| [HNSW_ACE_HALF](NEIGHBORS_ANN_HNSW_ACE_HALF_UINT32_TEST.md) | 16 | 27.4 | 39 | 6%⁵ | ACE sub-builds 39%; half hnswlib search 19% | `npartitions` dedupe ≈ 4.5 c; half SIMD ≈ 3–4.5 c |
| [HNSW_ACE_FLOAT](NEIGHBORS_ANN_HNSW_ACE_FLOAT_UINT32_TEST.md) | 12 | 35.9 | 41 | ~4% | ACE sub-builds 59% | `npartitions` dedupe ≈ 3.5 c |
| [HNSW_ACE_INT8](NEIGHBORS_ANN_HNSW_ACE_INT8_UINT32_TEST.md) | 11 | 21.4 | 39 | 8%⁵ | ACE sub-builds 46% | `npartitions` dedupe ≈ 2.9 c |
| [HNSW_ACE_UINT8](NEIGHBORS_ANN_HNSW_ACE_UINT8_UINT32_TEST.md) | 11 | 18.4 | 39 | 10%⁵ | ACE sub-builds 42% | `npartitions` dedupe ≈ 2.9 c |
| [SCANN](NEIGHBORS_ANN_SCANN_TEST.md) | 10 | 13.1 | 105 | 20–23% | per-subspace PQ k-means 52%, coarse k-means 31% | library batched PQ k-means ≈ 4 n |
| [BRUTE_FORCE](NEIGHBORS_ANN_BRUTE_FORCE_TEST.md) | 5 | 6.5 | 78 | 16–20% | file I/O 32%, `CUFileInit` 30%, zero-fill 13% | no zero-fill 0.84 n |

¹ ≤ 15 blocks per launch; its idea 2 (`max_fraction` 0.1, ≈ 75 c) changes the configuration. ² 288 `DISABLED_` + 48 skipped; 580 cases
build. ³ Logged ctest times range from 15.6 to 86.6 s. ⁴ Warm JIT cache; savings refer to the 35.6 s cold ctest run. ⁵ Main `AnnHnswAceBuild` group.

## 2. Cross-cutting themes (ranked by total estimated savings)

Per-executable figures use each analysis's own unit (`n` or `c`). Totals are in `c` and standalone; they overlap across themes. "After T1"
means after the test-side reuse in T1.

### T1. Test-side index reuse; delete duplicate and ignored-parameter cases (≈ 745 c, keeps coverage)

Many cases rebuild an index identical to an earlier one, differing only in search parameters, output index type, or parameters the
fixture ignores. Some are exact duplicates.

* **IVF_PQ ≈ 180 c.** 868 of 1,397 cases rebuild an already-built key, and 332 are exact duplicates. The duplicates are the default-equivalent
  `enum_variety()` entries (`ann_ivf_pq.cuh:989,1012,1021,1026,1050`), plus u08 instantiating both `enum_variety()` and `enum_variety_l2()`
  (`test_uint8_t_int64_t.cu:18,23`). A static cache in `run()` (`:594`, `:795`, `:270`) saves 298 n; deleting only the duplicates saves 121 n ≈ 74 c.
* **CAGRA float ≈ 171 c, half ≈ 110 c, uint8 ≈ 73 c, int8 ≈ 77 c.** There are four sources of redundant builds: `_U32`/`_I64` build identical indices
  (float `.cu:16-17,26-27`, half `.cu:13-14,20-21`); AnnCagraTest ignores `merge_strategy`, `host_dataset`, `itopk_size` and `search_width`
  (`ann_cagra.cuh:451-455,467-476`); PHYSICAL/LOGICAL twins rebuild both halves; MultiPartition does 48 builds for 13 sets (`.cuh:2153-2173`). A
  process-wide cache saves 185 / 131 / 60 / 54–68 n (float / half / uint8 / int8). For float and half, the no-cache folds alone save 157 / 114 n.
  MultiPartition reuse adds 15 / 11 / 13 / 16 n.
* **IVF_FLAT ≈ 67–85 c.** Each case trains 3× on the same data (`ann_ivf_flat.cuh:126,274,448`); training once saves 88 n. A cache saves 76 n (≈ 20
  after A+B+E). The duplicates at `:539` and `:541` cost 5 n.
* **IVF_SQ ≈ 17 c.** `checkExtend` re-trains build 1 (`ann_ivf_sq.cuh:132`): reusing its quantizer saves 12.0 n, and a cache adds 9.8 n.
* **HNSW_ACE ×4 ≈ 13.8 c.** `npartitions` 0/1/2 all resolve to 2 (`cagra_build.cuh:1132-1141`), so 24 of 32 cases are triplicates.
* **Others.** BBQ ≈ 11.5: 108 builds for 27 graphs, and 27 dense builds for 3. TEST_BUGS ≈ 10 c: the #438 reproducer builds a 1,183,514 × 100 graph
  for a 6 ms search, and a random graph still triggers the bug (`bug_multi_cta_crash.cu:28-35`). RaBitQ ≈ 4.6; FILTER_UDF 1.3.
* **Effort and risk.** Effort S (folds, dedupes) to M (caches, ~60 lines). Risk L–M: a failing build fails every case that shares it; gtest filtering
  lowers the hit rate (clear the cache in `TearDownTestSuite`); builds are already non-reproducible (static `i_primes`); RaBitQ's `search` takes a
  non-const `index&` (`ivf_rabitq.cu:191`). Every search, serialize, merge and check still runs.

### T2. Library balanced k-means and per-subspace PQ training (≈ 525–595 c standalone; design estimates)

* **What it is.** One EM iteration costs 133–170 µs of wall time against 31–55 µs of GPU time. Each iteration: makes ~11 launches and ~6 alloc/free
  pairs; recomputes the norms (`kmeans_balanced.cuh:542`); does a blocking D2H + sync + host sort in `adjust_centers` (`:811-812`), even for the
  Random donor; allocates unused buffers (`:846`). PQ training repeats this serially for each subspace (`ivf_pq_build.cuh:349-415`, `:458`; ScaNN
  `pq.cuh:130-147`).
* **Fixes.** (A) One batched EM loop for all subspaces. (E) For the Random donor, skip the copy and sort; compute norms once; reuse workspaces.
* **Savings.** IVF_PQ: A ≈ 300–350 c, E ≈ 50–100 c (they overlap; neither is quantified after B); CAGRA float/uint8/int8/half: 80 / 35 / 34 / 55 n,
  assuming half of the pool is launch overhead; 35 / 15 / 13 / 12 n after T1; IVF_FLAT 45–65 n (15–25 after B); IVF_SQ 8–12 n (4–6 after D); RaBitQ
  2.5–3.5; ScaNN ≈ 4 n. Outside IVF-PQ, after T1: ≈ 85–100 c.
* **Effort and risk.** Effort L (A) / M (E). Risk M: codebooks and recall shift. Keeps coverage, and also helps users with a large `pq_dim` or VPQ.

### T3. CAGRA per-query search fallback for unaligned dims (≈ 170 c standalone, ≈ 155 after T1)

* **What it is.** Batched search needs a 16 B query row pitch (`cagra_search.cuh:94-114`): float `dim % 4`, half `% 8`, int8/uint8 `% 16`. Otherwise
  every query gets its own plan (`:114-160`): SINGLE_CTA launches 1-block grids; MULTI_KERNEL runs 100 queries × ~256 iterations × (4 kernels + a
  blocking `terminate_flag` D2H + sync), at `search_multi_kernel.cuh:586-592`; iterative builds and AddNodes hit the same path.
* **Fix.** Pass a query leading dimension into `setup_workspace` (`jit_lto_kernels/setup_workspace_impl.cuh:53,143`) and the
  multi-kernel/random_pickup kernels, then drop the loop.
* **Savings.** float ≈ 54 n (45 after T1); uint8 ≈ 50 n; int8 ≈ 51 n (16 of it dim 8); half ≈ 30 n (23 after T1).
* **Cheaper options.** Polling `terminate_flag` every N iterations saves 10–14 / 11 / 5 n (uint8 / int8 / half). Test-only, FilterTest dim 8 → 16
  (`.cuh:2017,2036`) saves ≈ 13 n each for uint8 and int8, with a slight change in coverage.
* **Effort and risk.** Effort M (the JIT fragment signatures change). Risk M: the fallback guards against misaligned reads
  (`cagra_search.cuh:95-100`). Keeps coverage.

### T4. Serialization overhead (≈ 130 c)

* **(a) Zero-fill, ≈ 58 c (≈ 45 after T1).** `kvikio_ofstream` zero-fills a 32 MiB `std::vector<char>` on every open
  (`cpp/src/util/file_io.cpp:265,403`; size at `include/cuvs/util/file_io.hpp:466`). This is inferred from a 10–33 ms untraced gap. Fix:
  `make_unique_for_overwrite`, or lazy growth. Effort S, risk L. Savings: VAMANA 27; CAGRA float 10, half 9.4, uint8 5.5, int8 4.3 n; IVF_SQ 1.5 n;
  RaBitQ 1.2; BRUTE_FORCE 0.84 n; HNSW ×4 2–2.5 c.
* **(b) Per-list blocking IVF I/O, ≈ 64 c.** Each list costs a header `pwrite`, 2 blocking device writes and a sync (`ivf_list.cuh:115-129,200-222`),
  plus ≈ 45 ms of cuFile register/deregister per case. Fix: one pinned staging copy and a few large writes, with the format unchanged. Effort M, risk
  M. Savings: IVF_FLAT 70 n (53 c), IVF_SQ 14 n (11 c). IVF-PQ uses the same path but was not costed.
* **(c) `CUFileInit`, ≈ 14 c; reduces GDS coverage.** It costs 0.93–1.15 s per process, because `open_kvikio_file_for_device_io` tries
  `CompatMode::OFF` first (`cpp/src/util/kvikio_io.hpp:54-72`). Fix: honour `KVIKIO_COMPAT_MODE=ON`, set it for the gtests, and keep one GDS job.
  Savings: ≈ 1 c in each of the 14 executables that pay it (extrapolated).

### T5. Slow test verification (≈ 125 c, keeps coverage)

* **(a) `ann_utils.cuh` helpers, ≈ 80 c.** `calc_recall` (`:228-280`) is O(n_q·k²), and its second, index-only pass is discarded by both callers
  (`:295`, `ann_cagra.cuh:1067`); `check_unique_indices` (`:164-193`) builds a `std::set` per row. Fix: sort each row and binary-search in one pass,
  and use `adjacent_find`. Effort S, risk L. Savings: RaBitQ 39.5; NN_DESCENT 24 n ≈ 18 c; IVF_PQ 21–28 n ≈ 15 c; IVF_FLAT 11 n ≈ 8 c.
* **(b) Sync-heavy checks, ≈ 44 c.** IVF_FLAT's packer check (`ann_ivf_flat.cuh:301-389`) does ~3 syncs and 5 D2H copies per list: ≈ 40 n; IVF_PQ's
  `compare_vectors_l2` (`ann_ivf_pq.cuh:100-125`) does ~1.9 k managed reads per case, each a D2H + sync: 20 n; the adaptive-centre check costs 3 n.

### T6. NN-descent host loop (≈ 37–51 c quantified)

* **What it is.** Every iteration spawns and joins a `std::thread` (`nn_descent.cuh:2754/2789`; BBQ `2900/2917`). 36-thread OpenMP regions, in two
  pools, run over only 2–10 k rows (`:2077, 2167, 2203, 2239, 2282, 2302, 2814, 2850`). Each build makes ~9 pinned alloc/free pairs. An iteration
  takes 2.3–5.6 ms, against 0.3–1.1 ms of GPU work.
* **Fix.** Use a persistent worker, size the OpenMP teams by `nrow`, and pool the pinned buffers. Before changing code, try `OMP_NUM_THREADS=1/4/8`
  and `KMP_BLOCKTIME=0`.
* **Savings.** NN_DESCENT 20–35 n ≈ 15–26 c (+2 n for pinned buffers); CAGRA_INT8 up to 15 n (≈ 8 after T1); BBQ 5–8 n (≈ 1.5 after T1). HNSW-ACE (29%
  of the float profile) and the other CAGRA executables were not costed.
* **Effort and risk.** Effort S–M. Risk L–M: keep full parallelism for large `nrow`. Keeps coverage. The fork/join attribution is inferred.

### T7. `graph_core` reverse-graph loop for host graphs (≈ 38 c, ≈ 20 after T1)

* **What it is.** For each column of a host graph: an OpenMP gather, an H2D copy, a kernel and a sync (`graph_core.cuh:836-852`).
* **Fix.** Copy the graph to the device once and reuse the device path (`:828-834`). Effort S, risk L, keeps coverage.
* **Savings.** CAGRA float 13 (7 after T1), uint8 6 (3), int8 8 (4), half 10 n (5); BBQ 0.6 n; HNSW-ACE float 1–2.4, half 1.5, int8 0.9, uint8 0.75 c.
  Every ACE sub-build takes this path, and identical work took 1.05–6.4 s across the HNSW profiles.

### T8. JIT linking and the CUDA ComputeCache (≈ 30 c with a cold cache)

* **What it is.** nvJitLink output is cached in `~/.nv/ComputeCache`. The local cache holds 1,073,442,295 B in 2,239 entries: it is at the 1 GiB
  default `CUDA_CACHE_MAXSIZE` and evicts LRU. CI starts with a cold cache. *The attribution is inferred:* cache-file mtimes match the ends of links
  and stalls, and no cold run was traced.
* **FILTER_UDF.** The fragment key contains the UDF source (`sample_filter_udf.cuh:78-86`), so every new UDF re-links: ≈ 2 s per (source, dtype), 72%
  of the test. A persisted cache saves ≈ 10.8; one parameterised UDF (`test_filter_udf.cu:52-84`) saves ≈ 6 and keeps coverage; library split-compile
  might save up to ~10 (not measured).
* **TEST_BUGS.** 4 iterative cases at `n_dim` 1024 stall ~5 s each: 20.2 s cold, ≈ 0 warm.
* **CAGRA ×4.** JIT costs 32.5 / 25.3 / 15.5 / 14.4 n, mostly because each group ran in its own process (est. 3–7 s in a single process). Not counted.
* **Fix.** Persist `CUDA_CACHE_PATH` across CI jobs, keyed on the libcuvs build, and raise `CUDA_CACHE_MAXSIZE` (e.g. 4 GiB). Effort S–M, risk L.

### T9. Smaller library items (≈ 60 c, grab bag)

* **IVF.** `recompute_internal_state` (`ivf_common.cuh:267-273`) makes 2·n_lists single-pointer H2D copies: IVF_FLAT 12, IVF_PQ 6, IVF_SQ 1.8 n. The
  IVF-Flat pack/unpack kernels run one thread per row (`ivf_flat_helpers.cuh:56-126`): ≈ 20 n. The IVF-PQ encode kernel takes 127 ms per launch at
  `pq_dim` 3072 (`ivf_pq_build.cuh:682`): ≈ 26 n.
* **Vamana.** Per-batch host syncs (`vamana_build.cuh:434-436,474`) cost 4–7 s. GreedySearch uses one warp per insert and ≤ 15 blocks (`:331,341`);
  fixing it is worth "potentially most of" 225 n, but it is high effort and not in the total.
* **HNSW and the rest.** Half SIMD in `cpp/cmake/patches/hnswlib.diff`: 3–4.5 c; brute-force upper HNSW layers (`hnsw.hpp:486-487`, FIXME): ≈ 2.5 c;
  one allocation per query in refine (`ivf_flat_build.cuh:489-491`): ≈ 1.7; ScaNN AVQ syncs (`scann_avq.cuh:614-628`): ≈ 1 n; RaBitQ throwaway
  rotators: ≈ 0.6.

### T10. Vamana reverse-batch over-allocation and `PERCENT 100` (suite-level, not measured)

* **What it is.** `max_reverse_batch = reverse_batchsize` (default 1e6, `vamana.hpp:77`) sizes `rev_ids`/`rev_dists` at 1e6 × visited_size × 4 B
  (`vamana_build.cuh:253,303-306`). On 1,000-row data that is an 8.5 GB peak (computed from the code), even though ≤ N rows are used.
* **Fix.** Clamp to N (`:253`, `:201`, `:496`) and lower `PERCENT 100` to ~25 (`cpp/tests/CMakeLists.txt:309`). The direct saving is only ≈ 0.25 s per
  process. The real gain is that other tests can share the GPU during Vamana's ~317 s, and it fixes a user-facing over-allocation.
* **Other tests.** NN_DESCENT (`:300-301`, < 100 MB) and BRUTE_FORCE (`:214`, ~1.4 GB) also reserve the whole GPU. While writing this summary I
  noticed that most entries in that file use `PERCENT 100`.

## 3. Options that reduce coverage or change the test configuration

**All of these need GPU re-validation of the listed thresholds.** Savings are standalone unless noted.

| executable | change | est. savings | re-validate on GPU |
|---|---|---|---|
| IVF_PQ | `kmeans_n_iters` 20 → 10 (`ann_ivf_pq.cuh:39-42`), or → 5 for big dims only | 35–40% (≈ 175–200 c); ≈ 125 c | `min_recall` 0.86, big-dims formula `:930` |
| IVF_PQ | *reduces coverage:* big dims in one build path + the filter test; drop dim 6144 (+ near-dups 513/1023/1025/2049/2050) | 218 n ≈ 133 c; 110 n ≈ 67 c (+122 n) | whether 6144 has its own search path |
| VAMANA | `max_fraction` 0.06 → 0.1 for deg 64/128/256 (`ann_vamana.cuh:335,354,373`) | ≈ 75 (43 after the trim) | CheckGraph at deg 256 |
| VAMANA | *reduces coverage:* deg 64/128/256 dims → {1, 8, 64, 137, 384, 619, 1024}; deg-32 2⁴ grid → resolution-IV half fraction | 92; 59. Ideas 1+2+3+4: 323 → 115 s | — |
| IVF_FLAT | `kmeans_n_iters` = 5; `nlist` 1024 → 128 for high dims (`:544-559,624-625`) | 90 n (30 after B); 85 n (20 after A+E) | cases 51/66 (0.77), radix (0.5), 40/128 |
| IVF_FLAT | *reduces coverage:* `stringstream` round trip instead of a kvikio file | 65–80 n | — |
| IVF_SQ | `nlist` 256; `kmeans_n_iters` = 10; *reduces coverage:* trim big dims | 21 n (8 after A+D); 12 n; 5.2 n | `nprobe/nlist` 0.04 → 0.16 |
| NN_DESCENT | `max_iterations` 100 → 20; *reduces coverage:* `n_rows` {2000} + host_dataset subset | 27 n ≈ 20 c; 72.7 n ≈ 53 c | gd = 64 recall, BBQ 0.15–0.35 |
| CAGRA_BBQ | `max_iterations` ~10; *reduces coverage:* GraphOnlyBuild on 1–3 params; 27 → 15 params | 5.3 n (1.1 after T1); 2.3; 1.2 | `min_recall_ratio`, > 0.8 baseline |
| CAGRA ×4 | *reduces coverage:* IVF_PQ only for SINGLE_CTA in the corner-case block (the T1 cache recovers most of it) | float 33, uint8/int8 18, half 35 n | — |
| CAGRA ×3 | FilterTest MULTI_KERNEL with 10 queries (or `itopk` 64) | uint8/int8 ≈ 22, float ≈ 10 | 0.995 recall at `itopk` 64 |
| IVF_RABITQ | `kmeans_n_iters` = 5; *reduces coverage:* k 16384 → 2048 (moot after T5); search-only params on one path; drop 2050 | 5; 38; 3.9; 0.87 | `n_probes` 1/2/4, 1-bit margins |
| SCANN | k-means 24 → 10 and PQ 10 → 5 iterations; *reduces coverage:* host-input TEST_Ps on 7 of 35 params; drop dim 2048 | 3.5–4.5 n; 5.3 n ≈ 5 c; 3.4 n | reconstruction thresholds `ann_scann.cuh:254-269` |
| TEST_BUGS | iterative `n_dim` 1024 → 128 (loses the 512-wide descriptor); extreme inputs 1e5 → 1e4 rows | ≤ 20 cold, ~0 warm; 1.7–2.5 | the throw still comes from `graph_core.cuh:1530` |
| BRUTE_FORCE / UDF / HNSW | round trip only ≤ 8 MiB; UDF tests on MULTI_KERNEL only; fallback test with ~9 partitions | 1.9 n; ~5.5; 0.45–0.9 c each | host limit still forces disk mode |


## 4. Notable findings beyond speed

1. **CAGRA half never runs dims ≥ 256.** `kBitshiftBase` = 11 makes SetUp skip them (`ann_cagra.cuh:172-176,210-212`): 218 of 1,986 cases. Dims
   128–255 get only 4 values per coordinate.
2. **FilteredMergeTest never sets `graph_degree`** (`ann_cagra.cuh:1263-1289`), so its degree sweep only produces 20 duplicates. Likely a bug.
3. **AnnCagraTest ignores four axes** (`merge_strategy`, `host_dataset`, `itopk_size`, `search_width`). The U32/I64 TEST_Ps differ only in output
   index type.
4. **ScaNN's `host_input_overlap` never overlaps:** 4096 rows fit in one batch (`scann_build.cuh:200-201`). A case with > 65,536 rows would cost
   ~0.3–0.5 s.
5. **The extreme-inputs test accepts any exception** (`bug_extreme_inputs_oob.cu:38`). Assert the message or the throw site (`graph_core.cuh:1530`).
6. **The #438 MultiCTA test checks no outputs.** #1818 is only caught reliably under compute-sanitizer, and needs `n_samples` > ~7,710.
7. **hnswlib half rounds every difference to fp16** via host `cuda_fp16` operators (`hnsw.hpp:234-241`). It is slow and also less accurate.
8. **`calc_recall`'s index-only second pass is discarded** by both callers (`ann_utils.cuh:295`, `ann_cagra.cuh:1067`).
9. **HNSW-ACE inputs with no effect.** `npartitions` 0/1/2 build identically. The fallback test hard-codes its limits (`ann_hnsw_ace.cuh:423-424`), so
   the inputs at `:1001-1002` are dead.
10. **Vamana's deg-256 rows have no recall check** (`ann_vamana.cuh:186`), although they are 43% of its runtime.
11. **NN_DESCENT.** All 288 UI8 cases are `DISABLED_` (`test_uint8_t_uint32_t.cu:16`). Termination scales with `dataset_dim`
    (`nn_descent.cuh:2740-2741`), which may be unintended. The fixtures use 100 iterations against a library default of 20.
12. **IVF builds are not reproducible across cases** (static `i_primes`, `kmeans_balanced.cuh:805`). IVF_SQ trains on the whole DB (256·`n_lists` >
    rows).
13. **Exact duplicates.** IVF_FLAT `:539` and `:541`; IVF_SQ `:449` = `:409`; RaBitQ cases 25/31/40 = case 0, and 44 = 29. RaBitQ dims 2049 and 2050
    both pad to 2112.
14. **Minor.** RaBitQ deserialize builds throwaway rotators (`ivf_gpu.cuh:161`, `ivf_gpu.cu:145`). The BBQ code-cache key has no data seed. Two
    slowdowns are unexplained: int8 IP/Cosine merges take ~5 ms per NN-descent iteration, and half IVF-PQ is 1.3–1.9× slower at dims 64/192.

## 5. Suggested order of work (rough cumulative `c`, using "after X" figures where given)

1. **Test-only, XS–S effort, no shared state: ≈ 665 c (~27%).** Dedupes: IVF_PQ 74, HNSW 13.8; CAGRA folds and axis filters: float 134, half 89, uint8
   36, int8 32; MultiPartition 50; `ann_utils` helpers 80; sync-free checks 44; IVF_FLAT train once 67; IVF_SQ quantizer 9; MultiCTA graph 10; BBQ
   11.5; one UDF source 6; Vamana `max_queries` 5. Also persist the JIT cache in CI (≈ 30 c cold), and try `OMP_NUM_THREADS`.
2. **Static index caches (M effort): +220 → ≈ 885 c (~36%).** IVF_PQ +106; CAGRA float +24, half +13, uint8 +24, int8 +22–36; IVF_FLAT +15; IVF_SQ +7;
   RaBitQ +3.6.
3. **Low-risk library fixes: +115 → ≈ 1,000 c (~40%).** Zero-fill 45, reverse graph 20, `recompute_internal_state` 14, NN-descent 26–37, hnswlib half
   3–4.5, HNSW upper layers 2.5. Also: the Vamana clamp + `PERCENT`, and optionally `KVIKIO_COMPAT_MODE` (≈ 14; this loses GDS coverage).
4. **Larger library work: +350 → ≈ 1,350 c (~55%).** CAGRA unaligned batching 155 (stop-gap: `terminate_flag` polling), batched IVF I/O 64, k-means/PQ
   outside IVF-PQ 85–100, IVF kernels 31. Excludes IVF-PQ batched training after B (300–350 standalone) and Vamana occupancy.
5. **Then the section 3 options**, after GPU re-validation.

## 6. Method and caveats

* **Profiling.** `nsys profile -t cuda,nvtx,osrt`, one fixture group / TEST_P / shard per process (2–35 per executable), on a 36-thread host. There
  was no CPU sampling (`perf_event_paranoid=4`), no cuVS NVTX ranges, and no memory tracing.
* **Inferred attribution.** Phases come from kernel markers, CUDA/OSRT gaps, gtest durations and log timestamps. Host-side attributions are inferred
  from timing and code: zero-fill, `calc_recall`, fork/join, JIT/ComputeCache. Vamana's 8.5 GB peak is computed from code.
* **nsys overhead.** Tracing inflates launch- and sync-bound phases (IVF_PQ has 47.2 M launches). For most executables nsys totals are 0.6–0.86× of
  ctest. CAGRA uint8/int8 and BBQ were slower under ctest (265 / 251 vs 208; 23 vs 16.4), so their `c` equals `n` here. Concurrent profiling added
  noise: ±40% for RaBitQ and BBQ, and identical HNSW work took 1.05–6.4 s.
* **Per-process artifacts.** One process per group repeats `CUFileInit` (~1 s), start-up (~0.5 s) and JIT (14–33 s per CAGRA executable). These are
  excluded from all savings. Keep each executable in one process.
* **Unvalidated.** No GPU runs or builds were allowed. Library estimates assume the fixed code runs at today's aligned or device-path cost. Ideas
  overlap within an executable, theme totals are not additive, and section 5 only roughly de-overlaps them. The scaling is crude.
* **Reference times.** Single `ctest -j8` runs; FILTER_UDF varied from 15.6 to 86.6 s, and TEST_BUGS' 35.6 s had a cold JIT cache.
* **Raw data.** `/tmp/claude-1000/-home-coder/ae252f72-9cc0-46ee-a543-8482b2c783fe/scratchpad/nsys_groups/<EXE>/*.nsys-rep`, plus per-group
  `*_sum.csv` files, gtest JSON and logs.
