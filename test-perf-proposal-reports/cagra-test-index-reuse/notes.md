# cagra-test-index-reuse: notes

Test-only change. It stops the CAGRA test executables `NEIGHBORS_ANN_CAGRA_{FLOAT,HALF,INT8,UINT8}_UINT32_TEST` from
rebuilding identical indices, and from instantiating cases a fixture cannot tell apart. Every distinct combination of
behaviour that was verified before is still verified with the same checks. The one intended change in what is
tested is the FilteredMerge `graph_degree` fix (section 4).

Files: `cpp/tests/neighbors/ann_cagra.cuh` (`.cuh`) and `cpp/tests/neighbors/ann_cagra/test_{float,half,int8_t,uint8_t}_uint32_t.cu`.
Line numbers refer to the patched files.

## 1. What changed

### Shared helpers (`.cuh`)

* `test_index_cache<KeyT, DataT, ValueT>` (`.cuh:358-420`) is a per-test-suite cache of the indices a fixture
  builds.
  * `get_or_build(res, key, dataset, size, build)` reuses an entry only if the key matches *and* the case's freshly
    generated dataset is bitwise identical to the entry's own copy (`thrust::equal` on the bytes). An incomplete key
    therefore cannot make a case search an index built over other data.
  * On a miss it copies the dataset into the entry and builds over that copy, so the index's dataset views stay
    valid after the fixture frees `database`.
  * If the build throws, nothing is cached.
  * The fixtures only read cached values: search, `serialize`, `cagra::merge` and `composite_index::search` don't
    modify their inputs (merge: see the `merge` docs in `cagra.hpp` and `merge_rebuild` /
    `append_to_input_graphs`).
  * Each fixture clears its cache in `TearDownTestSuite`.
* `cagra_build_key` / `make_cagra_build_key` (`.cuh:422-445`) is the build key. It holds `n_rows`, `dim`,
  `metric`, `graph_degree`, `build_algo`, `ivf_pq_search_refine_ratio`, `host_dataset` and `use_source_indices`.
  These are all the inputs that `index_params` and the build steps are derived from. The dataset is checked
  separately, as described above.
* `cagra_merge_halves` / `build_cagra_merge_halves` (`.cuh:447-497`) build the two half-indices (a 55 / 45 split).
  This code was duplicated in IndexMerge and FilteredMerge and is now shared.
* `unique_inputs`, `generate_cagra_test_inputs`, `generate_index_merge_inputs`, and the new lists
  `inputs_cagra_test` / `inputs_index_merge` are at `.cuh:2247-2330`. `inputs` is unchanged.

### Per fixture

| fixture | change (`.cuh` lines) |
|---|---|
| `AnnCagraTest` | `testCagra<SearchIdxT = IdxT, MoreSearchIdxT...>()` (:511-623) builds, or fetches from `build_cache()` (:728-743), applies `use_source_indices`, and does the serialize / deserialize round trip once. It then calls `searchAndCheck<T>()` (:625-709, the old naive_knn + search + `eval_neighbours` + `eval_distances` code, unchanged) once per output index type. Instantiated with `inputs_cagra_test`. |
| `AnnCagraFilterTest` | Builds through `build_cache()` (:1065-1098, :1337-1352). Everything after the build is unchanged. |
| `AnnCagraIndexFilteredMergeTest` | **Sets `index_params.graph_degree = ps.graph_degree` and `intermediate_graph_degree = 2 * graph_degree` (:1464-1465). Applies the degree-based recall relaxation that the other fixtures use (:1533-1536).** The halves come from `halves_cache()` (:1491-1497, :1580-1589). Still instantiated with `inputs`. |
| `AnnCagraIndexMergeTest` | `testCagra<SearchIdxT = IdxT, MoreSearchIdxT...>()` (:1610-1734) gets the halves from `halves_cache()` (:1692-1701, :1830-1839). It merges once (PHYSICAL) or builds the composite (LOGICAL), then calls `searchAndCheck<T>(search)` (:1736-1811, old checks unchanged) for each output index type. `testCagra<uint32_t>()` still works for `test_merge_fastener.cu`. Instantiated with `inputs_index_merge`. |
| `AnnCagraMultiPartitionTest` | `buildPartitions` (:2425-2463) gets the partition set from `partition_cache()` (:2730-2751), keyed by (`n_rows`, `dim`, `num_partitions`, `split`, `metric`, `build_algo`). The `part_padded_` member is gone. Search and FilteredSearch share the 13 partition sets. |
| `AnnCagraAddNodesTest` | Unchanged: `extend` modifies the index, so it can't be shared. |

### Test files

* `test_float_uint32_t.cu:15-17, 25-30, 39, 48` and `test_half_uint32_t.cu:12-14, 19-24, 26, 32`:
  * `AnnCagra_U32` + `AnnCagra_I64` become `AnnCagra_U32_I64`, which calls `testCagra<uint32_t, int64_t>()`.
  * `AnnCagraIndexMerge_U32` + `_I64` become `AnnCagraIndexMerge_U32_I64`.
  * AnnCagraTest is instantiated with `inputs_cagra_test`, and IndexMerge with `inputs_index_merge`.
* `test_int8_t_uint32_t.cu:21, 30` and `test_uint8_t_uint32_t.cu:21, 30`: the same two `ValuesIn` changes.

## 2. Why each removal keeps coverage

* **U32 / I64 fold.** The two TEST_Ps built, serialized and merged identical indices. Only the output index type of
  `search` differed. The folded body still runs naive_knn, `search`, `eval_neighbours` and `eval_distances` for
  *both* index types, on the same index (AnnCagraTest) or the same merged / composite index (IndexMerge).
* **`inputs_cagra_test`: 459 → 280.** AnnCagraTest never reads `merge_strategy`, `physical_merge_params` or
  `search_width`. It never sets `search_params.itopk_size` (it searches with the default). `host_dataset` only adds a
  D2H copy into a host matrix that is used only if `graph_build_params` holds `ace_params`, and no input selects ACE.
  So the `{PHYSICAL, LOGICAL}` axis of the corner-case, dim, team-size and n_rows blocks (72 + 84 + 15 + 2), and the
  `host_dataset {false, true}` axis of the refinement block (6), produced 179 exact duplicates. The first occurrence
  is kept.
* **`inputs_index_merge`: 459 → 453.** IndexMerge doesn't read `include_serialized_dataset`, `use_source_indices`,
  `search_width` or `smem_dtype`, and `host_dataset` is a no-op as above. Only the 6 `host_dataset` duplicates of the
  refinement block collapse. PHYSICAL / LOGICAL pairs stay separate cases (names stay stable), and they share the
  halves through the cache.
* **Caches.** A case only skips the build. Its search, serialize round trip, merge, filters and checks still run
  against an index built from the same parameters and from bitwise-identical data. Repeated builds could only catch
  nondeterministic build failures. Builds aren't bit-reproducible anyway (see `../PROFILING_SUMMARY.md` finding 12).

## 3. Case and build counts

Computed by replaying `generate_inputs()` and the fixtures' skip conditions (including the half `dim ≥ 256` and
uint8-only Hamming skips). The "before" executed counts match the analyses (float 2520 (1064), half 1986 (872),
int8 1128 (382), uint8 1128 (321)).

### Per fixture (instantiated / executed; "builds" = index builds; merge halves counted as pairs)

| fixture | executable | cases before | cases after | executed before | executed after | builds before → after |
|---|---|---|---|---|---|---|
| AnnCagraTest | float | 918 (2 TEST_Ps) | 280 | 596 | 190 | 596 → 107 (serialize round trips 596 → 190) |
| | half | 918 (2 TEST_Ps) | 280 | 540 | 176 | 540 → 93 (round trips 540 → 176) |
| | int8 | 459 | 280 | 298 | 190 | 298 → 107 |
| | uint8 | 459 | 280 | 321 | 203 | 321 → 117 |
| AnnCagraIndexMergeTest | float | 918 (2 TEST_Ps) | 453 | 524 | 256 | half-pairs 524 → 101; PHYSICAL merges 346 → 167 |
| | half | 918 (2 TEST_Ps) | 453 | 468 | 228 | pairs 468 → 87; merges 318 → 153 |
| | int8 | 459 | 453 | 262 | 256 | pairs 262 → 101; merges 173 → 167 |
| | uint8 | 459 | 453 | 285 | 279 | pairs 285 → 111; merges 186 → 180 |
| AnnCagraIndexFilteredMergeTest | float | 459 | 459 | 135 | 135 | pairs 135 → 80; filtered merges 135 → 135 |
| AnnCagraFilterTest | float, int8, uint8 | 60 | 60 | 60 | 60 | 60 → 13 |
| AnnCagraMultiPartitionTest | float, int8, uint8 | 48 | 48 | 48 | 48 | sets 48 → 13 (partition indices 252 → 67) |
| | half | 48 | 48 | 46 | 46 | sets 46 → 12 (244 → 63) |
| AnnCagraAddNodesTest | all | 102 | 102 | unchanged | unchanged | unchanged |

Where the executed counts drop, the removed cases are exact duplicates: AnnCagraTest 108 / 94 / 108 / 118 per
TEST_P, and IndexMerge 6 per TEST_P (float / half / int8 / uint8).

### Per executable (`--gtest_list_tests` count, executed = not skipped)

| executable | cases before | cases after | executed before | executed after |
|---|---|---|---|---|
| NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST | 2520 | 1417 | 1456 | 782 |
| NEIGHBORS_ANN_CAGRA_HALF_UINT32_TEST | 1986 | 883 | 1114 | 510 |
| NEIGHBORS_ANN_CAGRA_INT8_UINT32_TEST | 1128 | 943 | 746 | 632 |
| NEIGHBORS_ANN_CAGRA_UINT8_UINT32_TEST | 1128 | 943 | 807 | 683 |

Float also contains 15 non-parameterized tests (CagraQ*, MultiPartition rejects), which are unchanged.

Test names:
* `AnnCagra_U32` and `AnnCagra_I64` become `AnnCagra_U32_I64`, and `AnnCagraIndexMerge_U32` and `_I64` become
  `AnnCagraIndexMerge_U32_I64` (float, half). The filter `*AnnCagra_U32*` still matches.
* AnnCagraTest parameter indices are unchanged up to `/86` and shift after that.
* IndexMerge indices are unchanged up to `/449`.
* FilteredMerge, Filter, AddNodes and MultiPartition names are unchanged.

## 4. FilteredMerge `graph_degree`: wired in

The old code (`.cuh:1263-1289` before this change) never set `graph_degree` / `intermediate_graph_degree`, so every
case built degree-64 graphs (intermediate 128). The `{32, 47, 64}` sweep of `inputs` produced 20 duplicates, and the
printed `degree=` was wrong. `git log` shows that this was an oversight and not intended:
* PR #819 ("Adds test cases for different graph degrees", merged 2026-07-23) added `graph_degree` to `AnnCagraInputs`.
  It wired the field into AnnCagraTest, AddNodes, Filter and IndexMerge, together with the
  `degree < 50 → ×0.94, < 40 → ×0.94` recall relaxation.
* FilteredMerge had been added by #1496 (2026-01-29) and already existed in #819's parent commit, but #819's diff
  doesn't touch it.

This change does what #819 did for the other fixtures: it sets both degrees from `ps.graph_degree` and applies the same
relaxation as its sibling IndexMerge.

Effect on the 135 executed float cases:
* degree-64 cases (40): unchanged.
* degree-47 cases (20): now build degree 47 / 94 and require 0.935 instead of 0.995.
* degree-32 cases (75): now build degree 32 / 64 and require 0.879 instead of 0.995.

No duplicates remain. NN-descent size limits only get looser: smaller degrees, and `n_rows ≥ 500`.

Alternative, if the parent prefers not to change what is tested: keep the fixture as is and give it a deduplicated
list without the `graph_degree` axis. That is 20 fewer executed cases, with the same checks as before.

## 5. Expected effect

The estimates below come from the per-executable analyses (nsys seconds, so the absolute values are inflated). They
are for the analyses' "index cache" + MultiPartition ideas, which this change implements:
* float: ~185 s (1e) + ~15 s (MP) of 366 s.
* half: ~131 s + ~11 s of 257 s.
* int8: ~54-68 s (1c, includes FilterTest) + ~16 s of 208 s.
* uint8: ~60 s + ~13 s of 208 s.

In ctest terms, roughly 40-55 % for float and half, and 35-40 % for int8 and uint8. FilteredMerge's degree-32 / 47
builds are cheaper than the old degree-64 builds, which may save a little more. No GPU run was possible here, so these
figures are unmeasured.

## 6. Risks

* **Shared state.** A cached index is shared by every case with the same key. A bad build fails all of them, but it
  would have been rebuilt identically for each of them anyway. Mitigations:
  * strict keys;
  * the bitwise dataset check;
  * values that are never modified (`AnnCagraTest` passes the cached index only to `serialize`, and searches the
    deserialized copy);
  * a failed build is never cached.
* **GPU memory held during a suite.** It is bounded by the distinct builds of one suite and freed in
  `TearDownTestSuite`.
  * The largest is MultiPartition with float, about 100 MB: 13 sets, of which the dim-1024 set is about 35 MB.
  * AnnCagraTest, IndexMerge and FilteredMerge are each well under 50 MB, because their datasets are ≤ 1000 rows.
  * The executables run with `PERCENT 100`.
* **Static destruction.** The caches are function-local statics. They are empty by the time the process exits,
  because `TearDownTestSuite` always runs. A suite that is filtered out never creates its cache.
* **FilteredMerge degree change.** It changes the test (section 4). The degree-32 / 47 recall at the relaxed
  thresholds is unvalidated on a GPU.
* **Test name changes** (section 3) may affect dashboards that track individual names.
* **gtest filtering / sharding.** The cache just misses. The cost per case is then the same as before.

## 7. How to verify

1. Build the four executables plus `NEIGHBORS_ANN_CAGRA_MERGE_TEST`. `test_merge_fastener.cu` reuses
   `AnnCagraIndexMergeTest::testCagra<uint32_t>()`.
2. Check the case counts with `--gtest_list_tests`. The expected counts are the "after" values in section 3: 1417 /
   883 / 943 / 943.
3. Run each executable in one process. All tests should pass, and the skipped counts should be 635 / 373 / 311 / 260
   (float / half / int8 / uint8). Pay attention to `AnnCagraIndexFilteredMergeTest/*` (section 4).
4. Compare wall time and ctest time with the base branch.
5. Optionally, profile with nsys and count IVF-PQ / NN-descent builds. For example, AnnCagraTest float should show
   107 builds and IndexMerge float 101 half-pairs.

Compile check (done here, no GPU):
* The command came from `compile_commands.json`, with `-arch=sm_89`, `-Werror` and the object file in `$TMPDIR`.
* These TUs compile with no errors or warnings:
  * `ann_cagra/test_{float,half,int8_t,uint8_t}_uint32_t.cu`
  * `ann_cagra/test_merge_fastener.cu`
  * `ann_hnsw_ace/test_float_uint32_t.cu`
  * `ann_cagra/test_bbq_uint32_t.cu`
  * `ann_cagra/test_filter_udf.cu`
  * `ann_cagra/bug_multi_cta_crash.cu`
