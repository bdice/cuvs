### Train each IVF-Flat test index once and reuse it across sub-tests and cases

Every case of `NEIGHBORS_ANN_IVF_FLAT_TEST` trained three indexes on the same data, one each in `testIVFFlat`,
`testPacker` and `testFilter`. Cases that differ only in search parameters rebuilt and rechecked identical indexes.
This PR shares those builds:

* `testIVFFlat` keeps the trained empty index. `testPacker` and `testFilter` share `extend(database, no indices,
  trained)`, which is what `build` with `add_data_on_build = true` does. Pack/unpack do not depend on the trainset
  fraction or adaptive centres. `testPacker` now packs into a fresh empty index (`helpers::reset_index`) instead of
  mutating the trained one, so no shared index is modified.
* The two exact duplicate entries (`/12`, `/14`) set the new `independent_builds` flag and keep the old per-sub-test
  builds. This keeps `kmeans_trainset_fraction = 1.0` and the in-place `add_data_on_build` path covered for all four
  data types, with and without adaptive centres.
* A per-suite LRU cache (≤ 256 MiB of device memory, cleared in `TearDownTestSuite`) keeps the deserialized and the
  fully extended index of cases that passed all checks. It is keyed on `(num_db_vecs, dim, nlist, metric,
  adaptive_centers, host_dataset, kernel_copy_overlapping)`; the data type is per fixture. Later cases with the same
  key skip the build-only checks (file round trip, centroid invariants, packer). They still run their naive references,
  both searches and both recall checks.

All assertions and thresholds are unchanged. Test names and the case count (390) do not change. Per data type,
k-means trainings go from 291 to ≈ 76, and file round trips and packer loops from 97 to ≈ 72. Only
`cpp/tests/neighbors/ann_ivf_flat.cuh` changes.

## Measurements

Single-process wall time of each test executable on an RTX 6000 Ada (48 GB) with a 36-core host, otherwise idle (no ctest parallelism, no MPS). Old and new binaries were run alternately, 2 repetitions each with the order reversed between repetitions; mean (min–max).

| executable | tests before | tests after | before | after | change |
|---|---|---|---|---|---|
| `NEIGHBORS_ANN_IVF_FLAT_TEST` | 390 | 390 | 232.9 s (231.8–234.0) | 156.6 s (152.9–160.2) | -32.8% |

All tests passed in every run.

Peak GPU memory (RMM peak via `GTEST_CUVS_MEMORY_PEAK=1`; NVML process peak):

| executable | RMM before | RMM after | NVML before | NVML after |
|---|---|---|---|---|
| `NEIGHBORS_ANN_IVF_FLAT_TEST` | 1441 MiB | 1756 MiB | 1916 MiB | 2236 MiB |

Closes #`<issue>`
