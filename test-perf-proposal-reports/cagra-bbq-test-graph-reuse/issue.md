### [TEST] CAGRA BBQ tests build the same graph 4 times per parameter set

**Problem**

`NEIGHBORS_ANN_CAGRA_BBQ_UINT32_TEST` (`cpp/tests/neighbors/ann_cagra/test_bbq_uint32_t.cu`, fixture
`cpp/tests/neighbors/ann_cagra_bbq.cuh`) runs 5 TEST_Ps over 27 parameter sets: 9 code layouts × 3 metrics, each with
4000 × 128 rows and graph degree 32. About 86% of its run time is NN-descent graph construction plus `optimize`.
Most of those builds are repeats:

* `AnnCagraBbqSearchRecall`, `AnnCagraBbqGraphShape`, `AnnCagraBbqSerializeRoundTrip` and
  `AnnCagraBbqGraphOnlyBuild` each call `cagra::build` on the same codes with the same `default_index_params()`.
  `GraphOnlyBuild` differs only in `attach_dataset_on_build = false`, and that flag is applied after the graph is built
  (`cagra_build.cuh:2986-2992`). This is **108 BBQ builds for 27 distinct graphs**.
* `AnnCagraBbqSearchRecall` also builds a full-precision reference graph for every parameter set. That reference
  depends only on the data and the metric, not on the code layout. This is **27 dense builds for 3 distinct graphs**.

In total there are 135 NN-descent builds where 30 would do. In a profile on an RTX 6000 Ada, the executable took
16.4 s of gtest time under nsys and 14.0 s in a plain run. The redundant builds account for about 70% of that.

**Proposal**

This is a test-only change that keeps every check:

* Build each BBQ graph once per parameter set, with `attach_dataset_on_build = false`. Keep a host copy of the graph in
  a suite-level cache, cleared in `TearDownTestSuite`. `GraphOnlyBuild` asserts on that build's result.
  `SearchRecall`, `GraphShape` and `SerializeRoundTrip` each bind fresh codes to the cached graph through the same
  `index(res, metric, dataset, graph)` constructor the build uses for the default attach path. Each then searches,
  moves or serializes its own index.
* Compute the dense reference recall once per (seed, shape, k, degree, metric), and reuse it across the 9 layouts.
* Key both caches on everything the data and graphs depend on, including the data seed.

Test names, case count (135) and thresholds stay the same. The expected saving is about 9–11 s per run (≈ 70%).

**Note**

The CPU quantizer's on-disk code cache (`cpp/internal/cuvs_internal/preprocessing/bbq_cpu_quantize.hpp`, `cache_path`)
is keyed by n_rows, dim, layout and metric, but not by the data seed or a hash of the data. Two tests that share a
shape but use different data would load each other's codes. No current test collides, but the key should probably
include a data hash. That is separate from this change.
