### [TEST] IVF-PQ tests rebuild the same index for cases that differ only in search parameters

**Problem**

`NEIGHBORS_ANN_IVF_PQ_TEST` (`cpp/tests/neighbors/ann_ivf_pq/*.cu`, fixtures in `cpp/tests/neighbors/ann_ivf_pq.cuh`)
has 1,397 cases. It took 496 s under `ctest -j8` on an RTX 6000 Ada, and 811 s of gtest time under nsys. Each case
builds its own index and runs the per-list codepacking checks on it. PQ codebook training is 78 % of the time, and the
checks are another 11.5 %. Many of those builds are repeats:

* **332 cases are exact duplicates** of another case of the same TEST_P.
  * In `enum_variety()`, `{codebook_kind = PER_SUBSPACE}`, `{pq_bits = 8}`, `{force_random_rotation = false}`,
    `{lut_dtype = CUDA_R_32F}` and `{internal_distance_dtype = CUDA_R_32F}` all equal the default configuration. That
    is 5 identical entries in every metric block.
  * The uint8 tests instantiate both `enum_variety()` and `enum_variety_l2()`. They are identical, because L2Expanded
    is the default metric.
  * `defaults()` repeats the default entry of `enum_variety_l2()` with a weaker recall threshold.
* **Most of the remaining cases differ from an earlier case only in search parameters** (`k`, `n_probes`,
  `lut_dtype`, `internal_distance_dtype`, `coarse_search_dtype`). Examples are all 17 `var_k()` cases and 6 entries of
  every `enum_variety*` block. Each case still trains and checks a new index before its single search.
  `build_precomputed` doesn't search at all, so for that test such cases are pure repeats.

Overall, the 1,397 cases need only about 630 distinct builds.

**Proposal**

This is a test-only change:

1. Delete the exact duplicates. In every case that is removed, the remaining twin has the same parameters and an equal
   or stricter `min_recall`.
2. Keep the index built and checked by the previous case in a single-entry static cache, keyed by:
   * the build path (the TEST_P);
   * the data sizes, which fully determine the data because `gen_data()` uses a fixed seed;
   * every `ivf_pq::index_params` field.

   A case with the same key skips the build and the checks, and only runs its search and recall checks on the cached
   `const` index. That index is in the same state that case would have searched. `build_precomputed` skips such cases.
   An index that fails its checks is not cached. The cache is cleared on a key change and in `TearDownTestSuite`, so at
   most one index is held.
3. Order `enum_variety()` so that the search-only variants follow the default entry, which lets one entry serve them.

Every distinct (type, build path, build parameters, search parameters) combination still runs with the same checks.
Every distinct index is still built, extended, serialized and checked. Builds go from 1,397 to 631.

**Expected impact**

The analysis estimates 15 % for deleting the duplicates alone, and about 37 % (≈ 180 s of 496 s) for reuse with an
unbounded cache. The 110 big-dim cases are 41 % of the time and have unique keys, so they are unaffected. This change
should save roughly 30–37 %.

Trade-offs:

* Cases that share an index pass or fail together.
* The number of cases drops from 1,397 to 1,065.
* Index-based test names shift after removed entries.
* 24 `build_precomputed` cases report `SKIPPED`.
