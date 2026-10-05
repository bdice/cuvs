# NEIGHBORS_ANN_IVF_FLAT_TEST: where the time goes and how to make it faster

Source: 6 nsys runs, one per fixture group (`-t cuda,nvtx,osrt`, RTX 6000 Ada). All times below are gtest time under
nsys, where CUDA tracing inflates launch-bound code. The whole executable took **275 s** under `ctest -j8`, so
"≈ real" means s × 0.76 (275 / 361).

## Summary

| metric | value |
|---|---|
| cases / groups | 390 / 6: 4 × 97 `AnnIVFFlatTestF_{float,half,int8,uint8}.AnnIVFFlat` + 2 single TESTs (0.2 s each) |
| total gtest time (nsys) | **361.4 s** (float 98.3, half 93.1, uint8 85.1, int8 84.4) |
| GPU busy (kernels + copies + memsets) | **24–26 %** of the CUDA span per group (kernels: 19 s per group). The test is host-, sync- and I/O-bound. |
| kernel launches | **12.3 M** (≈ 31.7 k per case, mean kernel 6.5 µs). Per float group: 2.3 M `cudaMemcpyAsync`, 0.71 M stream syncs, 2.2 M pool alloc/free pairs |
| per-case time | flat: 0.2–1 s for most cases. **60 high-dim cases (dim 2048–2056, 4096) take 126.5 s (35 %)**, at 1.9–3.6 s each |
| structure | every case runs `testIVFFlat()` + `testPacker()` + `testFilter()` (`test_*_int64_t.cu:21-26`) → **3 k-means builds**, 4 extends, 1 file round trip, 2 naive references per case |

## Where the time goes

Each case was split into phases using kernel markers (`rngKernel`, `naive_distance_kernel`, k-means kernels,
`build_index_kernel`, `interleaved_scan`, `pack/unpack_interleaved_list_kernel`) and kvikio NVTX ranges. Totals are over the 4 big groups.

| phase | s (nsys) | % | GPU busy | code |
|---|---|---|---|---|
| **k-means training, 3× per case** | **≈ 135** | **37 %** | 32 % | `ann_ivf_flat.cuh:126` (47.2 s), `:274` (≈ 45.3 s), `:448` (≈ 42.5 s) → library `kmeans_balanced` |
| **index file serialize + deserialize** | **98.3** (60.1 + 38.2) | **27 %** | 5 % | `ann_ivf_flat.cuh:189-191` → `ivf_flat_serialize.cuh:74-76,154-156`, `ivf_list.cuh:115-129,200-222` |
| **packer check, per-list loop** | **72.8** | **20 %** | 40 % | `ann_ivf_flat.cuh:301-389` |
| search + adaptive-centre check + host recall (×2) | 25.8 | 7 % | 16 % | `:194-254`, `:464-487` |
| extends (4 per case) + packer resize/`recompute_internal_state` | ≈ 20 | 5.5 % | low | `:137,145,277,291-297` |
| naive references, SetUp, teardown | ≈ 9 | 2.5 % | 90 % (naive) | `naive_knn.cuh`, `:490-515` |

* **k-means: same launch-bound pattern as IVF-PQ** (see `NEIGHBORS_ANN_IVF_PQ_TEST.md`, idea E). With `n_lists = 1024`,
  `build_hierarchical` (`kmeans_balanced.cuh:1292`) trains 32 mesoclusters and then 32 fine-cluster sets, 20 EM
  iterations each: **≈ 680 EM iterations per build** (counted via `sum_rows_by_key` kernels), ≈ 790 k iterations in
  total. Each iteration costs ≈ 170 µs wall time but only ≈ 55 µs of GPU time. Per iteration there are ~11 launches, ~6
  alloc/free pairs, two norm reductions (dataset norms recomputed), and the 256-byte pageable D2H copy + sync in
  `adjust_centers` (`kmeans_balanced.cuh:811-812`). **Redundancy:** the three builds in a case train on the same
  data. `testIVFFlat` uses trainset fraction 0.5, `testPacker` uses 1.0 with adaptive off, and `testFilter` uses 0.5
  with `add_data_on_build` (which is just build + `extend(…, nullptr)`, `ivf_flat_build.cuh:436-438`).
* **Serialization is per-list, blocking I/O.** For every non-empty list (up to 1024), the kvikio path writes the
  numpy header into the staging buffer, syncs the stream (`kvikio_serialize.hpp:78`), and flushes the header with a
  blocking `pwrite` (`file_io.cpp:297`). It then writes data and indices with two separate blocking kvikio
  device writes. Each device write is a bounce-buffer `cuMemcpyBatchAsync` + `cuStreamSynchronize` + `pwrite`. Lists of
  ≥ 256 KB instead go through a `cuFileWrite` on the kvikio worker thread, and the main thread waits on a futex for
  it (10.7 s of futex per float group). Deserialize mirrors this per list: alloc, header parse, sync, two device `pread`s.
  Measured: ≈ 120 µs per list at dim 16 and ≈ 880 µs per list at dim 2048. At dim 2048 a list holds ~10 rows padded
  to 32, so 256 KB per list and 256 MB per index. Fixed cost is ≈ 45 ms per case: two cuFile handle register
  (≈ 5 ms) and deregister calls (≈ 15 ms each, up to 95 ms after large writes), plus `CompatModeManager`.
  `CUFileInit` costs 0.94 s once per process.
* **Packer loop.** For each list the test does 5 allocations, a gather, `pack`, a mask `map_offset`, `thrust::reduce`
  (sync + D2H, `:344`), 2 `map_offset`s, `devArrMatch` (`test_utils.cuh:46-68`: 2 pageable D2H copies + sync + host
  loop, `:370`), `unpack`, and `devArrMatch` again (`:383`). That is about 3 syncs and 5 D2H copies per list. Float
  totals: 321 k syncs, and most of the group's **18.4 GiB of D2H** traffic. The library kernels themselves are slow:
  `pack/unpack_interleaved_list_kernel` (`ivf_flat_helpers.cuh:56-84`) uses one thread per row that loops over all
  dims, with 1 block for lists ≤ 256 rows. That is 87 µs per launch at dim 2048 and 273 µs at dim 2049 (veclen 1), and
  **22.4 s of GPU time = 30 % of all kernel time** in this executable.
* **Smaller items.** `recompute_internal_state` (`ivf_common.cuh:267-273`) issues 2·n_lists 8-byte pageable H2D
  copies after every extend/deserialize (~8 calls per case). That is 1.59 M copies and 4.1 s of API time per group.
  The host recall `calc_recall` (`ann_utils.cuh:228-280`) is O(n_q·k²) and runs twice per eval. For the 1 M/100 k-query
  cases (`:628-633`) this shows up as 0.6 s GPU-idle gaps, ≈ 3.5 s per group. The adaptive-centre check
  (`:218-236`) runs copy_selected + mean + `devArrMatch` per list: 3.6 s in total.
* **Not hotspots:** process start (≈ 0.7 s to the first test), JIT/module load (`cuLibraryLoadData` 14 calls =
  44 ms; first search per metric ≈ 40 ms), `naive_knn` (5.4 s, GPU-bound), `interleaved_scan` (0.4–0.8 s per group),
  data generation (0.7 s). Only ≈ 8 s per group is in GPU-idle gaps > 20 ms; the rest is 5.8 M micro-gaps.
* **Redundant parameters.** Entries `:539` and `:541` exactly duplicate `:535` and `:537`. In each group, **28 of
  97 cases repeat an earlier case's index-build key** (`num_db_vecs, dim, nlist, metric, adaptive, host,
  overlap`) and differ only in `nprobe`/`num_queries`: 28/30/32/92, 29/31/33/75/94, 36/38, 37/39/79, the host and
  overlap triplets, 70/72, 71/73, 74/93, 95/96. Each repeat re-runs 3 builds, serialization and the packer check
  (76 s in total). In the high-dim block, 2049/2050/2051/2053 all use veclen 1 for float.

## Ideas (ranked by standalone estimated savings; % of the 361 s; ≈ real = × 0.76)

Savings overlap: B, D and C all shrink k-means; G overlaps A and E; F overlaps everything. The recommended combination
that keeps coverage is **B + A + E + H + I + K: ≈ 235 s (≈ 65 %, ≈ 180 s real)**. After that, add D or C (≈ 30 s)
and F (≈ 20 s).

**D. Fewer k-means iterations in the tests** *(keeps code paths, changes the configuration)*
* Change: set `index_params.kmeans_n_iters = 5` (default 20, `ivf_flat.hpp:32`) at `ann_ivf_flat.cuh:117,268,442`.
  The hierarchical EM count drops from ≈ 680 to ≈ 170 per build.
* Savings: ≈ 90 s (25 %, ≈ 68 s real) standalone, ≈ 30 s after B. Effort: S. Risk: L–M. Most `min_recall = nprobe/nlist`
  thresholds are ≤ 0.07, and cases with `nprobe = nlist` are independent of clustering, but the 51/66 cases (0.77) and the
  radix cases (0.5) must be revalidated on the GPU.

**B. Train once per case and share the index across the three sub-tests** *(keeps coverage of what is checked)*
* Change: build the empty index once (`:126`, or in `SetUp`). In `testPacker` reuse it instead of `:274`: the packer
  only needs an empty index plus `extend`, and centres and adaptivity don't affect packing. In `testFilter`, search
  `index_loaded`/`index_2` (already fully extended) or `extend(database, nullopt, idx)` instead of rebuilding
  with `add_data_on_build` (`:441-448`). Keep one dedicated case for `add_data_on_build = true` and fraction 1.0.
* Savings: two of three trainings, **≈ 88 s (24 %, ≈ 67 s real)**. Effort: S–M. Risk: L.

**G. Use fewer lists for the high-dim entries** *(keeps code paths, changes the configuration)*
* Change: `nlist 1024 → 128` at `:544-559` and `:624-625`. The shared-memory-limit search path depends on `dim`, not on
  `nlist`. Lists grow from ~10 rows (padded to 32) to ~80 rows, so there are 8× fewer per-list I/O calls, pack calls
  and checks, and 2.7× less padded data. Optionally also drop near-duplicate dims 2051/2053 *(reduces coverage)*.
* Savings: ≈ 85 s of the 126.5 s (24 %) standalone, ≈ 20 s after A + E. Effort: S. Risk: L. `min_recall` rises to
  40/128 and must be checked on the GPU.

**F. Cache the built and checked index across cases with the same build key** *(keeps coverage)*
* Change: a small static cache in the fixture, keyed by `(DataT, num_db_vecs, dim, nlist, metric, adaptive, host,
  overlap)`. On a hit, skip the build, extend, serialize and packer/adaptive checks, and only run the two
  searches + evals (the DB is identical because the seed is fixed and the DB is generated first). The same idea as
  IVF-PQ B. Also delete the exact duplicates at `:539` and `:541` (K).
* Savings: 28 of 97 cases per group → **≈ 76 s (21 %)** standalone, ≈ 20 s after A + B + E. Effort: M. Risk: L–M
  (state shared across cases).

**A. Batch IVF list (de)serialization in the library's kvikio file path** *(keeps coverage)*
* Change: in `ivf_flat::detail::serialize` (`ivf_flat_serialize.cuh:68-77`), copy all lists into one pinned host
  buffer with async copies and a single sync. Then emit headers and payloads through the staging buffer, so a few
  large `pwrite`s replace 4 blocking writes + 2 syncs per list (`ivf_list.cuh:115-129`, `kvikio_serialize.hpp:78-79`,
  `file_io.cpp:291-309`). In `deserialize_impl` (`:152-157`), read the list section in one `pread` and issue async H2D
  copies with one sync. Keep direct GDS for lists above a size threshold. The format is unchanged, and the change
  also removes the per-file cuFile handle register/deregister (≈ 45 ms per case). Helps every IVF index
  (IVF-PQ/SQ share `ivf_list.cuh`).
* Savings: **≈ 70 s (19 %, ≈ 53 s real)**. Effort: M. Risk: M (I/O path; large-index GDS behaviour).
* Test-only alternative *(reduces coverage of the kvikio file path)*: round-trip through `std::stringstream` (host-staged
  path, ≈ 1 sync per list) or through a file only in one case per metric. Savings ≈ 65–80 s.

**E. Make the packer check sync-free and fix the pack/unpack kernels** *(keeps coverage)*
* Test (`:301-389`): allocate `flat_codes`, mask and buffers once at the max list size. Replace `thrust::reduce` ==
  check and both `devArrMatch` calls with device-side compares that accumulate into one device error counter, read
  once after the loop. Savings ≈ 40 s.
* Library (`ivf_flat_helpers.cuh:56-126`): parallelise over (row, dim-chunk) with a 2-D grid instead of one thread per
  row. Savings ≈ 20 s of GPU time (users of `codepacker::pack/unpack` benefit too).
* Together: **≈ 58 s (16 %, ≈ 44 s real)**. Effort: S (test) + S–M (kernel). Risk: L.

**C. Cut per-iteration overhead in the library's balanced k-means** *(keeps coverage)*. Same changes as
IVF-PQ idea E: skip the D2H copy + sort in `adjust_centers` for the Random donor path (`kmeans_balanced.cuh:811-819`),
compute dataset norms once (`predict`, `:542-548`), and reuse EM workspaces. ≈ 45–65 s standalone (35–50 % of
k-means), ≈ 15–25 s after B. Effort: M. Risk: L–M.

**I. Batch the pointer copies in `recompute_internal_state`** (`ivf_common.cuh:267-273`) *(keeps coverage)*: fill
two host vectors and do 2 copies instead of 2·n_lists. ≈ 12 s (3 %). Effort: S. Risk: L. Shared with IVF-PQ K.

**H. Faster host recall** (`ann_utils.cuh:228-280`) *(keeps coverage)*: merge the two O(k²) passes into one, and
parallelise over rows (OpenMP) or compute on the GPU. ≈ 11 s (3 %), only in the 1 M/100 k-query cases. Effort: S. Risk: L.

**K. Delete the exact duplicate entries** `:539` and `:541` *(keeps coverage)*: 8 cases, ≈ 5 s (1.4 %). Effort: XS.

**J. Vectorise the adaptive-centre check** (`:218-236`) *(keeps coverage)*: recompute all centres with one
`calc_centers_and_sizes` on the index's labels and compare once. ≈ 3 s (1 %). Effort: S.

## Uncertainties

* **nsys inflation.** The groups sum to 361 s of gtest time under CUDA tracing, versus 275 s for the whole executable under
  contended `ctest -j8`. Launch- and sync-bound phases (k-means, per-list I/O, packer loop) are probably inflated more
  than GPU-bound ones, so real savings may be somewhat smaller than × 0.76 suggests.
* **Phase boundaries are inferred** from kernel markers (no cuVS NVTX ranges) and are accurate to a few ms per case.
  The k-means numbers include the label prediction of the following extend. The packer/filter split between k-means
  and extend uses the measured extend cost (≈ 3.4 s per call site).
* **Serialization cost attribution** comes from kvikio NVTX ranges and OSRT calls, plus code reading. The cuFile
  behaviour (compat mode, 15–95 ms deregistration) depends on this machine's filesystem and cuFile setup, and may differ in CI.
* **Not validated on the GPU** (nothing could be run): recall with `kmeans_n_iters = 5` (D) or `nlist = 128` (G), and the
  design estimates for A, C and E. Savings for B and F assume cached or shared indices behave like fresh ones. Builds
  already differ run to run because of the static `i_primes` in `adjust_centers`.
