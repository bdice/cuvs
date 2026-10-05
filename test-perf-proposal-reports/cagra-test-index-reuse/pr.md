### Reuse CAGRA test indices across cases and drop duplicate cases

The CAGRA gtests rebuilt an identical index in most of their cases. This PR changes only the tests
(`cpp/tests/neighbors/ann_cagra.cuh` and `ann_cagra/test_{float,half,int8_t,uint8_t}_uint32_t.cu`).

* **One body for the uint32 and int64 outputs** (float, half).
  * `AnnCagra_U32` + `_I64` become `AnnCagra_U32_I64`, and `AnnCagraIndexMerge_U32` + `_I64` become
    `AnnCagraIndexMerge_U32_I64`.
  * `testCagra<uint32_t, int64_t>()` builds, serializes and merges once. It then runs the naive_knn + search +
    `eval_neighbours` + `eval_distances` checks for each output index type.
  * `testCagra<T>()` still works, so `test_merge_fastener.cu` is unaffected.
* **No cases a fixture can't tell apart.**
  * AnnCagraTest uses `inputs_cagra_test` (459 → 280 entries). It ignores `merge_strategy`, `itopk_size` and
    `search_width`, and `host_dataset` is only an unused D2H copy.
  * IndexMerge uses `inputs_index_merge` (459 → 453 entries), which drops the `host_dataset` duplicates.
  * `inputs` itself is unchanged.
* **Index reuse.** `test_index_cache` is a per-test-suite cache.
  * It is keyed by every build parameter.
  * An entry is reused only if the case's generated dataset is bitwise identical to the copy it was built from.
  * Cached indices are only searched, serialized or merged. None of these modify them.
  * Each fixture clears its cache in `TearDownTestSuite`.
  * It is used by AnnCagraTest, FilterTest, IndexMerge and FilteredMerge (for the two halves) and by
    AnnCagraMultiPartitionTest (for the partition sets).
  * AddNodes isn't cached, because `extend` modifies the index.
* **FilteredMerge honours `graph_degree`.** It never set the degree, so every case built degree-64 graphs and the
  `{32, 47, 64}` sweep only produced duplicates. #819 wired the degree, and the degree-based recall relaxation, into
  the other fixtures but missed this one. It now does the same as IndexMerge.

Every distinct combination of parameters still runs with the same checks and thresholds. The only change in what is
tested is the FilteredMerge fix: its degree-32 / 47 cases now build degree-32 / 47 graphs.

## Measurements

Single-process wall time of each test executable on an RTX 6000 Ada (48 GB) with a 36-core host, otherwise idle (no ctest parallelism, no MPS). Old and new binaries were run alternately, 2 repetitions each with the order reversed between repetitions; mean (min–max).

Test counts are listed cases (including cases that skip in SetUp). `NEIGHBORS_ANN_CAGRA_MERGE_TEST` and `NEIGHBORS_ANN_CAGRA_FILTER_UDF_TEST`, which include the changed header but not the changed fixtures, were unchanged within noise (1.8 → 1.6 s and 3.0 → 2.9 s, excluding a cold-JIT-cache first run).

| executable | tests before | tests after | before | after | change |
|---|---|---|---|---|---|
| `NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST` | 2520 | 1417 | 193.5 s (191.1–196.0) | 88.3 s (88.2–88.5) | -54.4% |
| `NEIGHBORS_ANN_CAGRA_HALF_UINT32_TEST` | 1986 | 883 | 137.2 s (136.5–137.9) | 52.0 s (51.6–52.4) | -62.1% |
| `NEIGHBORS_ANN_CAGRA_INT8_UINT32_TEST` | 1128 | 943 | 117.9 s (117.2–118.7) | 78.0 s (77.0–79.0) | -33.9% |
| `NEIGHBORS_ANN_CAGRA_UINT8_UINT32_TEST` | 1128 | 943 | 120.3 s (117.0–123.5) | 79.1 s (79.0–79.2) | -34.2% |
| **total** | | | 569.0 s | 297.4 s | -47.7% |

All tests passed in every run.

Peak GPU memory (RMM peak via `GTEST_CUVS_MEMORY_PEAK=1`; NVML process peak):

| executable | RMM before | RMM after | NVML before | NVML after |
|---|---|---|---|---|
| `NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST` | 400 MiB | 396 MiB | 920 MiB | 920 MiB |
| `NEIGHBORS_ANN_CAGRA_HALF_UINT32_TEST` | 391 MiB | 395 MiB | 912 MiB | 910 MiB |
| `NEIGHBORS_ANN_CAGRA_INT8_UINT32_TEST` | 391 MiB | 395 MiB | 918 MiB | 918 MiB |
| `NEIGHBORS_ANN_CAGRA_UINT8_UINT32_TEST` | 391 MiB | 395 MiB | 922 MiB | 922 MiB |

Closes #`<issue>`
