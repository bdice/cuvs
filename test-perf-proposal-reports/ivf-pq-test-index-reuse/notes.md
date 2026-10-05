# ivf-pq-test-index-reuse

Test-only change to `NEIGHBORS_ANN_IVF_PQ_TEST`. It makes two changes:

* It deletes the 332 cases that repeat an earlier case of the same TEST_P exactly.
* It reuses the index that the previous case built and checked when the current case differs only in search parameters
  (`k`, `search_params`, `min_recall`).

Every distinct (data type, build path, build parameters, search parameters) combination that ran before still runs.
Each one keeps the same or a stricter recall threshold, and every distinct index is still built, extended, serialized
and checked by the per-list codepacking checks. Builds drop from 1,397 to 631.

Background: `../NEIGHBORS_ANN_IVF_PQ_TEST.md` ideas B and F, and T1 in `../PROFILING_SUMMARY.md`.

## What changed

All line numbers refer to the files after the change. They were formatted with clang-format 20.1.8 (`cpp/.clang-format`).

### `cpp/tests/neighbors/ann_ivf_pq.cuh`

* **`make_index_key(build_path, inputs)`** (`:147-176`). Returns a `std::tuple` of everything that determines the
  dataset and the built index:
  * `build_path` (a `std::type_index`), `num_db_vecs`, `num_queries` and `dim`;
  * every field of `ivf_pq::index_params`: `metric`, `metric_arg`, `n_lists`, `kmeans_n_iters`,
    `kmeans_trainset_fraction`, `pq_bits`, `pq_dim`, `codebook_kind`, `codes_layout`, `force_random_rotation`,
    `conservative_memory_allocation`, `add_data_on_build` and `max_train_points_per_pq_code`.

  The dataset comes from `gen_data()`, which uses the fixed seed 1234. For a given type, it depends only on
  `num_db_vecs × dim`, because the database is drawn before the queries. `num_queries` is in the key anyway. It costs
  no hits, since every instantiated case uses 1024. The data type is covered because the cache is a static member of
  the fixture class template, so each `<EvalT, DataT, IdxT>` has its own. The key leaves out only `k`, `min_recall` and
  `search_params`.
* **`last_index_cache<IdxT>`** (`:178-221`) is a single-entry cache holding an optional key and a
  `std::unique_ptr<index<IdxT>>`.
  * `lookup(key)` returns true on a hit. On a miss, it releases the cached index first, so that a new build never
    overlaps an old cached index.
  * `put(key, idx)` stores a key with its index. The index may be null; `build_precomputed` stores only a key.
  * `get()` returns the cached index, and `clear()` empties the cache.
* **`ivf_pq_test`**
  * `build_precomputed()` (`:353-410`). This test never searches, so a case that differs from the previous one only in
    search parameters would repeat it exactly.
    * The key uses a local tag type `build_precomputed_path` as the build path.
    * On a hit, the case calls `GTEST_SKIP()` with a message.
    * On success, it stores the key with no index (`:409`). The body is otherwise unchanged.
  * `get_compression_ratio()` and `check_lists()` (`:687-714`). The per-label loop (reconstruct/extend, packing,
    reconstruction) moved out of `run()` unchanged.
  * `run()` (`:716-...`) works in four steps:
    * **Key:** `make_index_key(typeid(BuildIndex), ps)`. Each TEST_P passes its own lambda, and each lambda
      expression has a unique closure type, so the type identifies both the TEST_P and the fixture.
    * **Miss:** `build_index()`, then `check_lists()`. The index is cached only if `!HasFailure()`. If the checks
      failed, this case still searches its own uncached index (as before), and the next case rebuilds.
    * **Hit:** the build and the checks are skipped.
    * **Search:** both paths then run the unchanged search, recall check and out-of-bounds checks on a `const index&`.
  * `static void TearDownTestSuite()` (`:828`) clears the cache. The static member is declared at `:839`.
* **`ivf_pq_filter_test`.** `run()` (`:919-...`) works the same way, with no checks between build and search. It also
  has `TearDownTestSuite()` (`:1001`) and its own static cache (`:1012`).
* **`enum_variety()`** (`:1112-1197`) goes from 17 to 13 entries:
  * The 5 entries equal to the default configuration become 1 entry, which sets those 5 values explicitly with
    `min_recall = 0.86`. The 5 entries were `{codebook_kind = PER_SUBSPACE}`, `{pq_bits = 8}`,
    `{force_random_rotation = false}`, `{lut_dtype = CUDA_R_32F}` and `{internal_distance_dtype = CUDA_R_32F}`, each
    with `min_recall = 0.86`.
  * That entry sits where `{force_random_rotation = false}`/`{lut_dtype = 32F}` were, directly before the 6
    search-only variants. All 7 cases with the default build key are now consecutive, so a single-entry cache serves
    them.
  * The relative order of all other entries is unchanged.

### Instantiations (`cpp/tests/neighbors/ann_ivf_pq/`)

* `test_float_int64_t.cu:19-21, 27-29` drops `defaults()`, in both `f32_f32_i64` and `f32_f32_i64_filter`.
* `test_int8_t_int64_t.cu:17-18, 21-22` drops `defaults()`, in both `f32_i08_i64` and `f32_i08_i64_filter`. The
  copyright year was updated.
* `test_uint8_t_int64_t.cu:17-19, 22-24` drops `enum_variety()` and keeps `enum_variety_l2()`, in both suites. The
  copyright year was updated.

## Removed cases, and why they were duplicates

All removals are within a TEST_P, and each removed case has a remaining twin in the same TEST_P. The counts below come
from a Python model of the parameter lists, which is not committed. For every TEST_P, the model checked two things:

* The set of distinct (build key, `k`, `search_params`) combinations is identical before and after. For
  `build_precomputed` the combination is the build key only, because that test ignores search parameters.
* For each combination, the remaining case's threshold is at least the strictest threshold among the old cases.

| removed | per TEST_P | why it is a duplicate |
|---|---|---|
| 4 of the 5 default-equivalent `enum_variety()` entries, in each metric block | 4 per block (f32: 4 blocks, i08: 3, u08: 4) | Identical `ivf_pq_inputs`: `codebook_kind`, `pq_bits`, `force_random_rotation`, `lut_dtype` and `internal_distance_dtype` are all at their defaults, and `min_recall` is the same, also after the `_ip`/`_cosine` scaling. |
| `enum_variety()` in u08 | 13 (after the line above; 17 before) | `enum_variety_l2()` only sets `metric = L2Expanded`, which is the default metric. The two blocks are identical. |
| `defaults()` in f32 and i08 | 1 | Same build and search parameters as the default entry of `enum_variety_l2()`. The only difference is `min_recall`: unset, which gives the heuristic `min(erfc(…), n_probes / n_lists) ≤ 20/32 = 0.625`, against 0.86 in the entry that stays. `eval_neighbours` fails iff `recall < min_recall - eps`, with the same `eps`, so the remaining case is strictly stricter. |

### Case counts (`--gtest_list_tests`)

| suite | TEST_Ps | cases per TEST_P, before → after | total before → after |
|---|---|---|---|
| `IvfPq/f32_f32_i64` | 5 (host_input, host_input_overlap, extend, serialize, precomputed) | 88 → 71 | 440 → 355 |
| `IvfPq/f32_f32_i64_flat_layout` | 1 | 4 → 4 | 4 → 4 |
| `IvfPq/f32_f32_i64_filter` | 1 (build_search) | 88 → 71 | 88 → 71 |
| `IvfPq/f32_i08_i64` | 4 (build, host_input, host_input_overlap, serialize) | 79 → 66 | 316 → 264 |
| `IvfPq/f32_i08_i64_filter` | 1 | 79 → 66 | 79 → 66 |
| `IvfPq/f32_u08_i64` | 4 (build, host_input, host_input_overlap, extend) | 94 → 61 | 376 → 244 |
| `IvfPq/f32_u08_i64_filter` | 1 | 94 → 61 | 94 → 61 |
| **total** | | | **1397 → 1065 (−332)** |

The total matches the analysis's count of 332 exact duplicates: 321 identical including `min_recall`, plus the 11
`defaults()` cases.

### Test names

Names are index-based (`IvfPq/<suite>.<test>/<i>`), so indices after a removed entry shift. Names stay stable where
possible: every surviving entry keeps its relative order. The mapping from new to old index:

* **f32:** `0–18` → `+1` (small_dims, big_dims_moderate_lut). Enum block `b = 0..3` starts at new `19 + 13b`, old
  `20 + 17b`.
* **i08:** `0–26` → `+1` (big_dims, var_k). Enum block `b = 0..2` starts at new `27 + 13b`, old `28 + 17b`.
* **u08:** `0–8` unchanged (small_dims_per_cluster). Enum block `b = 0..3` starts at new `9 + 13b`, old `26 + 17b`.
  The old block at `9–25`, `enum_variety()`, is gone.
* **Within an enum block, new → old position:**

  | new | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | old | 0 | 2 | 3 | 4 | 5 | 7 | 1, 6, 8, 9, 14 (merged) | 10 | 11 | 12 | 13 | 15 | 16 |

## Builds after the change

Builds counted from the parameter lists. A build is a cache miss, and in `build_precomputed` a case that is not skipped.

| group (per TEST_P) | cases | distinct build keys | builds | reuses |
|---|---|---|---|---|
| f32 (each of 5 TEST_Ps + filter) | 71 | 47 | 47 | 24 (in `build_precomputed` these 24 are `SKIPPED`) |
| i08 (each of 4 TEST_Ps + filter) | 66 | 31 | 32 | 34 |
| u08 (each of 4 TEST_Ps + filter) | 61 | 37 | 37 | 24 |
| flat_layout | 4 | 4 | 4 | 0 |
| **total** | **1065** | | **631** (was 1397) | **434** (410 searches on a reused index + 24 skips) |

* i08 builds the default key twice per TEST_P. `var_k()` (all default key) is followed by `enum_variety_l2()`, whose
  default entry comes after 6 build variants. Moving `var_k()` would save 5 small builds but would renumber 17 more
  names, so it was not done.
* By the parameter lists, 771 cases (not the 868 given in the analysis) have a build key that already appeared
  earlier in the same TEST_P. The analysis's keying could not be reproduced. All counts here come from the code.

## Cache design and memory

* **Size and eviction.** There is one entry per fixture class template instantiation. A miss releases the old index
  before building a new one. `TearDownTestSuite` releases it at the end of each suite, and gtest never interleaves
  suites. So at most **one** cached index is alive in the process.
* **Hit rate.** gtest runs all parameters of one TEST_P before the next TEST_P, so the parameter lists put the cases
  that share a key next to each other:
  * the 7 default-key entries of each `enum_variety*` block;
  * all 17 entries of `var_k()`.
* **Peak memory.** It differs from before in one way: the previous case's index stays alive during the next case's
  `SetUp` (`gen_data` + `naive_knn`), until that case's `run()` misses and releases it. The largest index is about
  170 MB (dim 6144), mostly the 6144² rotation matrix. The big-dim keys are all unique, so they never hit.
* **What is shared.**
  * The cached index is the index after `build_index()`, which includes `extend` for `build_extend_search` and
    serialize/deserialize for `build_serialize_search`, and after `check_lists()`, which rewrites the lists. That is
    exactly the state each old case searched.
  * A cached index is never checked again, so it is never mutated twice.
  * `search` takes `const index&`. Its only side effect is the lazily converted `centers_half_`, `centers_int8_` and
    `rotation_matrix_{half,int8}_` (`src/neighbors/ivf_pq_impl.hpp:90-94`). These are deterministic conversions of
    centers that never change after caching. A later case with the same `coarse_search_dtype` reuses them, where it
    used to compute them itself.
* **Streams and memory resources.** The index's buffers come from the current device resource on the handle's stream.
  For a default `raft::resources` that stream is `cudaStreamPerThread`, so they can be freed after the per-test handle
  is gone. The stream pool that `build_host_input_overlap_search` sets is used only for batch prefetch, not for index
  storage (`ivf_pq_build.cuh:1097-1103`). The index stores no handle.
* **Filtering and shuffling.** `--gtest_filter`, `--gtest_shuffle` and `--gtest_repeat` stay correct. Only the hit
  rate changes. A case run alone builds its own index.

## Expected effect

PQ training (78 %) and the per-label checks (11.5 %) now run 631 times instead of 1,397. The 110 big-dim cases are 41 %
of the time, and all of them still build because their keys are unique. Of the remaining 1,287 small cases, about 521
still build: 631 builds minus 110 big-dim builds. That suggests about 30–37 % less time for the executable, roughly
150–180 s of the 496 s under `ctest -j8`. The analysis estimated 37 % (≈ 180 s) for an unbounded cache and 15 % for
the duplicate removal alone. Measurements are still to be taken.

## Risks

* **Shared failures.** A bad build now fails every case that searches it: up to 7 cases for an enum key, 17 for
  `var_k`. Before, each case drew its own index. Builds were already not bit-reproducible across cases, because of the
  process-wide `static i_primes` in `adjust_centers`. An index that fails `check_lists` is not cached.
* **Recall statistics.** The search-only variants now search one index instead of 7–17 independent draws. A
  borderline threshold is hit by all of them together or by none, instead of independently.
* **24 `SKIPPED` cases** in `f32_f32_i64.build_precomputed` (6 per metric block). CI does not fail on skips. If a
  silent pass is preferred, replace `GTEST_SKIP()` with `return`.
* **Key maintenance.** A new `ivf_pq::index_params` field that a test varies must be added to `make_index_key`, as
  noted in its comment. Otherwise two different indices could share a key.
* **Name shifts.** Index-based names change as described above, which matters for any external filter or skip list
  that uses indices.

## How to verify

1. Build and list the tests:
   ```
   ./NEIGHBORS_ANN_IVF_PQ_TEST --gtest_list_tests | grep -c '^  '          # 1065 (was 1397)
   ./NEIGHBORS_ANN_IVF_PQ_TEST --gtest_list_tests | awk '/^[^ ]/{s=$1} /^  /{n[s]++} END{for(k in n) print k, n[k]}'
   # IvfPq/f32_f32_i64. 355, IvfPq/f32_f32_i64_flat_layout. 4, IvfPq/f32_f32_i64_filter. 71,
   # IvfPq/f32_i08_i64. 264, IvfPq/f32_i08_i64_filter. 66, IvfPq/f32_u08_i64. 244, IvfPq/f32_u08_i64_filter. 61
   ```
2. Run the full executable. Expect `[  PASSED  ] 1041 tests` and `[  SKIPPED ] 24 tests`, with all skips in
   `IvfPq/f32_f32_i64.build_precomputed/{26..31,39..44,52..57,65..70}`.
3. Check that a reused case also passes alone, e.g. `--gtest_filter='IvfPq/f32_i08_i64.build_search/24'` (`var_k`,
   k = 1023). It should build its own index.
4. Check the build count, optional. In an nsys run, the number of PQ-training phases (the markers used in the
   analysis) should drop from 1,397 to 631.
5. Time it: `ctest -R NEIGHBORS_ANN_IVF_PQ_TEST` before and after, or the executable alone.

Compile check: all three TUs compiled with the build's own nvcc command (`-arch=sm_89`, `-Werror` and
`-Werror=all-warnings`), with the output in `$TMPDIR`, without errors or warnings. The only output was the
environment's `NVCC_PREPEND_FLAGS` `-ccbin` notice.
