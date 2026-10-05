### Reuse IVF-PQ test indices across search-only variants and drop duplicate cases

`NEIGHBORS_ANN_IVF_PQ_TEST` built and checked a fresh index in every one of its 1,397 cases. Most of them differ from
the previous case only in search parameters, and 332 are exact duplicates. This PR changes only the tests:

* **Duplicates.**
  * `enum_variety()` merges its 5 default-equivalent entries into one. That entry is explicit and keeps
    `min_recall = 0.86`.
  * The uint8 instantiations drop `enum_variety()`, which is identical to `enum_variety_l2()`.
  * The float and int8 instantiations drop `defaults()`. It is the same configuration as the default entry of
    `enum_variety_l2()`, which has a stricter threshold.
* **Index reuse.**
  * `last_index_cache<IdxT>` is a single-entry static cache in `ivf_pq_test` and `ivf_pq_filter_test`.
  * It is keyed by `make_index_key()`: the build path (the TEST_P's lambda type), the data sizes, and every
    `index_params` field.
  * On a hit, `run()` skips the build and the per-list checks, and searches the cached index. That index already
    passed those checks and is in the state the case would have searched.
  * `build_precomputed` doesn't search, so it skips repeated keys with `GTEST_SKIP()`.
  * An index that fails its checks is not cached.
  * A miss releases the old index before building, and `TearDownTestSuite` clears the cache. At most one index is
    alive at a time.
* **Ordering.** The search-only variants in `enum_variety()` now follow the default entry, so consecutive cases share
  a key.

Every distinct (type, build path, build parameters, search parameters) combination still runs, with an equal or
stricter recall threshold. Every distinct index is still built, extended, serialized and checked.

Effect on the cases:

* The number of cases drops from 1,397 to 1,065.
* Index builds drop from 1,397 to 631.
* 24 `f32_f32_i64.build_precomputed` cases report `SKIPPED`.
* Names after a removed entry shift by their index.

## Measurements

Single-process wall time of each test executable on an RTX 6000 Ada (48 GB) with a 36-core host, otherwise idle (no ctest parallelism, no MPS). Old and new binaries were run alternately, 2 repetitions each with the order reversed between repetitions; mean (min–max).

The 1,065 tests after include 24 `build_precomputed` cases that are skipped because their index was already built and checked by an earlier case.

| executable | tests before | tests after | before | after | change |
|---|---|---|---|---|---|
| `NEIGHBORS_ANN_IVF_PQ_TEST` | 1397 | 1065 | 437.9 s (437.1–438.8) | 280.4 s (280.1–280.8) | -36.0% |

All tests passed in every run.

Peak GPU memory (RMM peak via `GTEST_CUVS_MEMORY_PEAK=1`; NVML process peak):

| executable | RMM before | RMM after | NVML before | NVML after |
|---|---|---|---|---|
| `NEIGHBORS_ANN_IVF_PQ_TEST` | 681 MiB | 681 MiB | 1216 MiB | 1248 MiB |

Closes #`<issue>`
