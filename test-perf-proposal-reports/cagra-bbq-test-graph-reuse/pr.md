### Build each CAGRA BBQ test graph once per parameter set

Four TEST_Ps in `NEIGHBORS_ANN_CAGRA_BBQ_UINT32_TEST` each rebuilt the same BBQ graph per parameter set (108 builds
for 27 graphs). The dense reference was rebuilt for every layout although it depends only on the metric (27 builds
for 3). This PR shares those builds:

* `bbq_build()` runs one `cagra::build` per parameter set, with `attach_dataset_on_build = false`. It caches a host
  copy of the graph and the graph-only index's properties. `AnnCagraBbqGraphOnlyBuild` asserts on those properties.
* `bbq_index()` gives `SearchRecall`, `GraphShape` and `SerializeRoundTrip` their own `device_bbq_index`. It binds
  freshly quantized codes to the cached graph through the constructor `build` uses for the default attach path. No
  index or device buffer is shared between tests.
* `dense_reference_recall()` builds the full-precision reference once per (seed, n_queries, n_rows, dim, k, degree,
  metric).
* Both caches are host-only (≈ 14 MB) and are cleared in `TearDownTestSuite`. The keys include the data seed, now a
  named constant shared with `SetUp`.

All checks and thresholds are unchanged and still run for all 27 parameter sets. Test names and the case count (135)
do not change. NN-descent builds go from 135 to 30. Only `cpp/tests/neighbors/ann_cagra_bbq.cuh` changes, plus a
comment in `test_bbq_uint32_t.cu`.

The attach branch of `build_from_bbq_dataset` (a single constructor call) is now exercised through that constructor
rather than through `build` itself. The 9 layouts of a metric now share one reference recall instead of drawing their
own.

## Measurements

Single-process wall time of each test executable on an RTX 6000 Ada (48 GB) with a 36-core host, otherwise idle (no ctest parallelism, no MPS). Old and new binaries were run alternately, 2 repetitions each with the order reversed between repetitions; mean (min–max).

| executable | tests before | tests after | before | after | change |
|---|---|---|---|---|---|
| `NEIGHBORS_ANN_CAGRA_BBQ_UINT32_TEST` | 135 | 135 | 10.6 s (10.4–10.9) | 3.5 s (3.0–4.0) | -67.2% |

All tests passed in every run.

Peak GPU memory (RMM peak via `GTEST_CUVS_MEMORY_PEAK=1`; NVML process peak):

| executable | RMM before | RMM after | NVML before | NVML after |
|---|---|---|---|---|
| `NEIGHBORS_ANN_CAGRA_BBQ_UINT32_TEST` | 10 MiB | 10 MiB | 494 MiB | 494 MiB |

Closes #`<issue>`
