# NEIGHBORS_ANN_IVF_SQ_TEST: where the time goes and how to make it faster

Source: 3 nsys runs, one per fixture group (`-t cuda,nvtx,osrt`, RTX 6000 Ada). All times are gtest time under nsys. The whole executable
took **45 s** under `ctest -j8`, so "≈ real" means s × 0.76 (45 / 59.4). The shared IVF root causes are analysed in
`NEIGHBORS_ANN_IVF_FLAT_TEST.md` (FLAT) and `NEIGHBORS_ANN_IVF_PQ_TEST.md` (PQ). This file only quantifies them here.

## Summary

| metric | value |
|---|---|
| cases / groups | 135 / 3: `AnnIVFSQTestF_float` 117 (50.7 s), `AnnIVFSQTestF_half` 17 (8.5 s), `ExtendInPlace…` TEST 1 (0.17 s) |
| total gtest time (nsys) | **59.4 s**. The nsys wall time of 168.6 s is mostly (≈ 109 s) nsys report generation, an artifact |
| GPU busy | **19 %** (float) / 11 % (half) of the CUDA span. Host-, sync- and I/O-bound |
| API volume | 1.89 M launches (≈ 14 k per case), 0.87 M `cudaMemcpyAsync`, 0.43 M stream syncs, 1.39 M pool alloc/free pairs |
| per case | median 0.39 s. **105 cases use `nlist = 1024` on a 10 k-row DB: 53.0 s (89 %)**. dim ≥ 2048: 9 cases, 9.1 s |
| structure (`ann_ivf_sq.cuh:61-82`) | per case: 2 naive k-NN, **2 full builds (2 k-means trainings)**, 2 extends, 1 kvikio file round trip, 5 searches |

## Where the time goes

Phases were split per case (all 134 parameterised cases) by kernel markers and kvikio `FileHandle` NVTX ranges. The markers are `rngKernel` ×4 per case,
`fused_column_minmax_kernel`, `encode_and_fill_kernel`, `ivf_sq_scan` and `naive_distance_kernel`.

| phase (both groups) | s (nsys) | % | GPU busy s | code |
|---|---|---|---|---|
| **k-means training, 2× per case** | **24.0** | **40.4** | 7.6 | `build_index` at `:64` and `:132` → `ivf_sq_build.cuh:483` |
| **serialize (kvikio file)** | **15.5** | **26.0** | 0.3 | `:108` → `ivf_sq_serialize.cuh:64-66` → `ivf_list.cuh:115-129` |
| **deserialize** | **8.6** | **14.4** | 0.4 | `:110` → `ivf_sq_serialize.cuh:140-141` → `ivf_list.cuh:200-222` |
| extends (in build 1 + `checkExtend`) | 4.8 | 8.1 | 0.5 | `ivf_sq_build.cuh:353-359`: per-list `resize_list` + `recompute_internal_state` |
| 5 searches + evals + index teardown | 5.1 | 8.6 | 0.9 | teardown is ≈ 4.6 k `cudaFreeAsync` per case |
| naive references ×2 + SetUp | 1.3 | 2.1 | 0.9 | `naive_knn.cuh` (`:273`, `:164`) |

* **k-means is launch-bound, as in PQ (idea E) and FLAT (idea C).** There are 168.8 k EM iterations: ≈ 725 per build at `nlist = 1024`, 191 at 64.
  Each costs **142 µs of wall time, but only 47 µs of GPU time**. Training uses the whole DB, because `256·n_lists` exceeds the row count (the log warns on every build).
  **Redundancy:** `checkExtend`'s `build_index(false)` (`:132`) repeats build 1's training on the same data with the same
  `RngState{137}` (`ivf_sq_build.cuh:461`). That is **12.05 s (20.3 %)**.
* **Serialization uses FLAT's per-list blocking path** (FLAT, idea A). A regression over the dim ≤ 256 cases gives **26 ms fixed + 155 µs per non-empty list**
  per round trip (91 µs write + 65 µs read, R² 0.87), over 98.6 k lists in total. That is ≈ 185 ms per `nlist = 1024` case. InnerProduct cases leave most lists empty
  (3–280 non-empty), so their round trip takes 10–60 ms. **New fixed cost:** every `serialize` stalls ≈ 10.5 ms between the file open and the first `pwrite`, with no traced calls.
  This is the zero-filled 32 MiB `std::vector<char>` staging buffer of `kvikio_ofstream` (`src/util/file_io.cpp:264`; size set in
  `include/cuvs/util/file_io.hpp:466`): 1.54 s in total. `CUFileInit` costs ≈ 1.0 s once per process (the 1 s idle gap in the first case). Dim 2048–4096: 4.4 s over 9 cases.
* **Extends** take ≈ 17 ms each at `nlist = 1024`: 1024 × `resize_list` (2 allocs + 1 fill kernel), plus 2·n_lists 8-byte H2D copies in
  `recompute_internal_state` (`ivf_common.cuh:267-273`). That is 584 k copies and 1.56 s of API time in float.
* **Not hotspots:** module loading (17 `cuLibraryLoadData`, 0.05 s), process start (≈ 0.5 s), `ivf_sq_scan` (0.8 s GPU), host recall (k ≤ 300).

## Ideas (ranked by standalone savings; nsys s and % of 59.4 s)

1. **B. `nlist = 256` instead of 1024 for the dim/metric/k/nprobe sweeps** *(changes the configuration; keeps code-path coverage)*. Change `ann_ivf_sq.cuh:388-453,
   472-485, 497-501, 531-537` and `inputs_half`, and keep ≈ 8 cases at 1024 (one per metric, plus host). The 96 such cases take 0.46 s each, against a measured 0.10–0.24 s at
   `nlist` 64–512. At 256, EM iterations drop from 725 to ≈ 360, the round trip from 185 to ≈ 65 ms, and each extend loses 18 ms. Lists also grow to ≈ 39 rows,
   so they span more than one 32-row group. **≈ 21 s (35 %, ≈ 16 s real)**; ≈ 8 s after A + D. Effort S. Risk M: the `nprobe/nlist` thresholds rise from 0.04 to 0.16, so recall needs validation.
2. **A. Batch IVF list (de)serialization in the library** *(keeps coverage)*: FLAT idea A. IVF-SQ uses the same `ivf::serialize_list/deserialize_list`.
   **≈ 14 s (24 %, ≈ 11 s real)** of the 15.3 s per-list cost. Effort M. Risk M.
3. **C. Cache indices across cases that differ only in search parameters** *(keeps coverage)*: PQ B / FLAT F. Key the cache on
   `(DataT, num_db_vecs, dim, nlist, metric, host_dataset)`. The DB is identical for the same `(num_db_vecs, dim)`, because `SetUp` (`:242-247`) draws it first from a fixed seed.
   Cache `idx`, `index_loaded` and the extended index. On a hit, run only the naive refs, 5 searches and evals. 99 unique keys cover 134 cases, so 35 are hits. Skip dim ≥ 512.
   **12.7 s (21 %, ≈ 9.7 s real)**; 9.8 s after D, ≈ 6 s after A + D. Effort M. Risk L–M (state shared across cases).
4. **D. Reuse build 1's quantizer in `checkExtend` instead of re-training** *(keeps coverage, except the trivial `add_data_on_build=false` branch)*.
   Pass `idx` in (`:80,130-133`). Construct `index<uint8_t> idx_empty(handle_, index_params, ps.dim)`, the same constructor `build` uses (`ivf_sq_build.cuh:456`).
   Then `raft::copy` the `centers()`, `sq_vmin()` and `sq_delta()`; `extend` computes the norms. **12.0 s (20 %, ≈ 9.2 s real)**. Effort S. Risk L.
5. **E. `index_params.kmeans_n_iters = 10`** in `build_index` (`:291-295`) *(changes the configuration)*: **≈ 12 s (20 %)**, ≈ 6 s after D. Effort XS.
   Risk L–M (the thresholds are lax, but must be validated).
6. **F. Cut balanced k-means per-iteration overhead in the library** *(keeps coverage)*: PQ idea E (`adjust_centers` D2H + sync + sort, norm
   recompute, alloc churn). Going from 142 to ≈ 80 µs per iteration saves **8–12 s (14–20 %)**; 4–6 s after D. Effort M. Risk L–M.
7. **G. Trim the big-dim cases** `:433-441` to 2048/L2, 2049/Cos, 2050/IP and 4096/L2 *(reduces coverage)*: **5.2 s (8.8 %)**. Effort XS. Risk L.
8. **H. Batch the pointer copies in `recompute_internal_state`** (`ivf_common.cuh:267-273`; FLAT I / PQ K) *(keeps coverage)*: **≈ 1.8 s (3 %)**. Effort S.
9. **I. Stop zero-filling the 32 MiB `kvikio_ofstream` buffer** (`file_io.cpp:264`): use `new char[cap]` or a smaller default (`file_io.hpp:466`)
   *(keeps coverage)*. **1.5 s (2.6 %)**; this also saves ≈ 10 ms per serialize in FLAT, PQ and others. Effort XS. Risk L.
10. **J. Delete the exact duplicate** `:449` (= `:409`) *(keeps coverage)*: **0.45 s (0.8 %)**. Effort XS.
11. **K. `KVIKIO_COMPAT_MODE=ON` for this ctest** to skip `CUFileInit` *(reduces cuFile-path coverage; cuFile already runs its POSIX fallback
    here)*: **≈ 1 s per process (≈ 2 %)**. Effort XS.

Recommended combination that keeps coverage: **D + I + J + H** (≈ 15.8 s, all small), then **A** (+14 s) and **C** (+≈ 6 s). Together that is ≈ 34 s
(≈ 58 %, ≈ 26 s of the 45 s real). B and E are optional further configuration changes.

## Uncertainties

* **nsys inflation:** 59.4 s under nsys versus 45 s real, which includes contention and start-up. Launch- and sync-bound phases (k-means, per-list I/O,
  extends) are probably inflated more than the ×0.76 suggests, so the real savings may be somewhat smaller.
* **Phase boundaries are inferred** from markers (the export has no cuvs-domain NVTX ranges) and are accurate to a few ms per case. The 10.5 ms serialize stall is attributed
  to the buffer zero-fill by elimination: there are no traced API/OS calls in the gap, and it falls after `handle_` in member-init order. CPU sampling was unavailable.
* **Not validated on the GPU:** recall for B and E. A and F are design estimates carried over from FLAT and PQ. C and D assume a reused or cached
  index behaves like a fresh build; builds are already not bit-reproducible (static `i_primes`, `kmeans_balanced.cuh:805`).
* **The serialization cost depends on the filesystem and cuFile setup** (POSIX fallback here; 15–70 ms deregistration after large writes), so CI may differ.
