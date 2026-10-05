# ivf-flat-test-index-reuse

Test-only change to `NEIGHBORS_ANN_IVF_FLAT_TEST`. Each case now trains its index once and shares it between
`testIVFFlat`, `testPacker` and `testFilter`. A small byte-bounded cache also lets a case reuse the indexes of an
earlier case with the same build parameters. Only `cpp/tests/neighbors/ann_ivf_flat.cuh` changes. The four
`ann_ivf_flat/test_*_int64_t.cu` files (TEST_P bodies, instantiations) and the UDF test are untouched.

## What changed (`cpp/tests/neighbors/ann_ivf_flat.cuh`, new line numbers)

| where | change |
|---|---|
| `:52`, `:62` | New input field `AnnIvfFlatInputs::independent_builds` (default `false`), printed by `operator<<`. |
| `:77-158` `testIVFFlat` | Calls `buildIndexes()` unless the case hit the cache. Then it searches the deserialized index (`loaded_`, or the cached copy) and runs the same `eval_neighbours` check. It frees `loaded_` early if the index is too large to cache. |
| `:165-279` `buildIndexes` | The old build, two-step extend, serialize/deserialize, `ASSERT_EQ(size)` and centroid invariant checks of `testIVFFlat`, moved unchanged. The trained, empty index is kept in `trained_` (`:176`), the deserialized one in `loaded_` (`:238`). |
| `:281-424` `testPacker` | Returns immediately on a cache hit (`:284`). Otherwise the reference index is `full_index()`, i.e. no training (`:306`). The pack target is a new empty `index` + `helpers::reset_index` (`:316-318`) instead of the trained index, whose centres were never used. With `independent_builds` it trains its own index as before (`trainset_fraction = 1.0`, non-adaptive, `:286-305`). The pack/unpack/mask checks (`:330-423`) are unchanged. |
| `:426-528` `testFilter` | Searches `full_index()` (`:488`) instead of `build(add_data_on_build = true)`. With `independent_builds` it builds as before (`:473-487`). The naive reference, bitset, filtered search and threshold are unchanged. |
| `:661-673` `full_index()` | `extend(database, no indices, trained_)`. This is exactly what `build` with `add_data_on_build = true` does: `ivf_flat_build.cuh:436-438` trains and then calls `detail::extend(&index, dataset, nullptr, n_rows)`. It is built once per case and shared by `testPacker` and `testFilter`. |
| `:549-559` `SetUp`, `:562-571` `TearDown`, `:573` `TearDownTestSuite` | Cache lookup (LRU, move to front), insert only if `!HasFailure()`, clear at the end of the suite. |
| `:582-660` | `build_key` = `(num_db_vecs, dim, nlist, metric, adaptive_centers, host_dataset, kernel_copy_overlapping)`. Also `built_indexes {key, loaded, full, bytes}`, `kIndexCacheBytes = 256 MiB`, `device_bytes()` (centres + list data + list indices) and `cacheIndexes()` (insert at the front, evict the LRU tail beyond the budget). |
| `:706-732` `inputs` | The exact duplicates (old `:539` = `:535`, old `:541` = `:537`; cases `/12`, `/14`) now set `independent_builds = true`. Their parameters are otherwise unchanged. |

### Why the key is complete

`SetUp` draws the database first from the fixed `RngState(1234)`. So the database depends only on
`(num_db_vecs, dim)` and `DataT`, and the cache is a static member of each `AnnIVFFlatTest<T, DataT, IdxT>`
instantiation, so one cache per data type. The indexes additionally depend on `nlist`, `metric`, `adaptive_centers`,
`host_dataset` (host vs device build/extend path) and `kernel_copy_overlapping` (prefetching host extend). All the
other index parameters are constants in the fixture. `num_queries`, `k` and `nprobe` only affect the queries,
searches and thresholds, which run for every case. Cases with `independent_builds` never look up or fill the cache.

## What each sub-test builds vs reuses

| sub-test | before (every case) | after, cache miss | after, cache hit | after, `/12` and `/14` (`independent_builds`) |
|---|---|---|---|---|
| testIVFFlat | k-means (fraction 0.5, `adaptive = ps`), 2 extends, file round trip, `size` check, centroid check, search + eval | same (the trained index is kept as `trained_`) | search + eval on the cached deserialized index | same as before |
| testPacker | k-means (fraction 1.0, non-adaptive), extend(all), resize the trained index's lists, per-list pack/mask/unpack checks | extend(all) of `trained_` → `full_` (no k-means); pack target = fresh empty index; same checks | skipped (passed for the cached `full`) | same as before (own k-means + extend) |
| testFilter | k-means (fraction 0.5, `adaptive = ps`) + `add_data_on_build`, filtered search + eval | filtered search + eval on `full_` (no build) | filtered search + eval on the cached `full` | same as before (own `add_data_on_build` build) |

No index is mutated after it is shared. `extend(…, const index&)` clones its input (`ivf_flat_build.cuh:374-383`).
`trained_` has no lists, so `index_2` and `full_` get freshly allocated lists. The old in-place mutation of the
trained index in `testPacker` (resize lists + `recompute_internal_state`) now happens on a private empty index.
The centroid check still compares `index_2` against the unmodified `trained_`.

Why sharing preserves what is checked:
* **testFilter.** The non-dedicated index is built exactly like the old one: same k-means parameters (fraction
  0.5, `adaptive_centers = ps`), then one extend with all rows and implicit ids `0..n-1`. The only difference is the
  library entry point: `clone` + `detail::extend` instead of the in-place `detail::extend` inside `build`. Cases `/12` and
  `/14` keep the in-place `add_data_on_build` path covered for every data type, both adaptive (`/14`) and
  non-adaptive (`/12`).
* **testPacker.** Pack/unpack only depend on the list assignment and the layout (dim, veclen). In `extend`,
  labels are predicted (`ivf_flat_build.cuh:218`) before adaptive centres are updated (`:234-253`). So neither the trainset
  fraction nor `adaptive_centers` changes what the packer checks. `/12` and `/14` keep `kmeans_trainset_fraction = 1.0`
  covered.

## Case counts

Unchanged: 97 cases per TEST_P × 4 (`AnnIVFFlatTestF_{float,half,int8,uint8}.AnnIVFFlat/0..96`) plus 2 plain TESTs,
for 390 in total. All test names are unchanged. The `GetParam()` printout gains a trailing `,0`/`,1` (`independent_builds`).

Work per data type (cache hits estimated from the index sizes; the exact count depends on the k-means list balance):

| | before | after |
|---|---|---|
| cache hits | 0 | ≈ 25 (float: /30-33, 38, 39, 44-47, 52, 53, 58-61, 66, 67, 72, 73, 75, 79, 93, 94, 96), 26 for half/int8/uint8 (+ /92) |
| k-means trainings | 291 | ≈ 76 (float) / 75: 1 per miss + 3 each for /12, /14 |
| extends | 388 | ≈ 218 / 215 |
| file serialize round trips, packer loops, centroid checks | 97 each | ≈ 72 / 71 |
| naive references, searches, `eval_neighbours` | 194 each | 194 each (unchanged) |

## Expected effect

From `NEIGHBORS_ANN_IVF_FLAT_TEST.md` (nsys, 361 s of gtest time; 275 s under `ctest -j8`):
* Training once per case (idea B): ≈ 86 s. That is 2 of 3 k-means runs in 95 of 97 cases per group, plus one extend.
* Cache hits (idea F after B): ≈ 0.43 s saved per hit (k-means, 3 extends, file round trip, packer loop), ×≈ 103 hits ≈ 45 s.
* Total ≈ 130 s of 361 s under nsys (≈ 36 %), roughly 100 s of the 275 s ctest time. The 60 high-dim cases (dim
  ≥ 2048) only gain from B: they are never repeated, and float indexes of that size exceed the cache budget.

Device memory: the cache holds ≤ 256 MiB per suite and is released in `TearDownTestSuite`. On a miss, a cacheable
`loaded_` (≤ 128 MiB) lives until `TearDown`. Larger ones are freed right after the search, so peak memory for the
high-dim cases is unchanged (2 full-size indexes at a time, as before). Expected peak increase ≤ ≈ 384 MiB.

## Risks

* **Cross-case state.** A hit case relies on checks done by an earlier case. Entries are only inserted if that case
  had no failure, and every miss (including any case run alone via `--gtest_filter`, or with `--gtest_shuffle`) runs
  the full set of checks. A wrong cached index would still fail the hit cases' searches.
* **Less random re-sampling.** Builds are not bit-reproducible (static `i_primes` in `adjust_centers`). Before,
  every repeated key re-ran k-means with fresh randomness and re-checked serialization/packing on it. Now each key
  is built and checked once per suite. The same parameter combinations are covered, but fewer random k-means outcomes are sampled.
* **add_data_on_build / fraction 1.0 coverage** drops from every case to `/12` and `/14` (Cosine, dim 5 non-adaptive and
  dim 8 adaptive, all 4 data types). The library difference between the two paths is in-place extend vs clone +
  extend.
* **Cached indexes outlive the fixture's `raft::resources`.** Their buffers come from the global current device
  resource and are stream-ordered on `cudaStreamPerThread`, which is the default stream of every `raft::resources`.
  Nothing is owned by the handle. The kernel-copy-overlap stream pool is only used for temporary batches.
* **Ordering contract.** `testPacker`/`testFilter` need `trained_` from `testIVFFlat` on a miss. The TEST_P bodies already call them in that order. A
  `RAFT_EXPECTS` reports a clear error otherwise.
* Recall thresholds, eps and all assertions are unchanged. The filtered search runs on the same kind of index as
  before, so no new recall risk is expected.

## How to verify

1. Build `NEIGHBORS_ANN_IVF_FLAT_TEST` and run it: all 390 tests pass.
2. Per-case times (`--gtest_print_time`, default): the hit cases listed above should drop to roughly the cost of
   two naive references + two searches. Misses should drop by about two k-means trainings and one extend.
3. Independence: run single hit cases alone, e.g. `--gtest_filter='*AnnIVFFlatTestF_float.AnnIVFFlat/31'`, and
   run `--gtest_shuffle --gtest_repeat=2`. Both must pass (the miss path runs all checks).
4. Dedicated cases: `--gtest_filter='*AnnIVFFlat/12:*AnnIVFFlat/14'` (the old 3-build path).
5. Memory: `GTEST_CUVS_MEMORY_PEAK=1 ./NEIGHBORS_ANN_IVF_FLAT_TEST` before/after. The peak should rise by ≤ ≈ 384 MiB.
6. Optional: `compute-sanitizer --tool memcheck` on a filter such as `*AnnIVFFlatTestF_float.AnnIVFFlat/2[89]:*AnnIVFFlatTestF_float.AnnIVFFlat/3[0-3]`, which covers a miss followed by hits.

Compile check: all four TUs (`test_{float,half,int8_t,uint8_t}_int64_t.cu`) compile with the
`compile_commands.json` flags (`-Werror=all-warnings`, `-Wall,-Werror`) and `-arch=sm_89`, with no warnings from the code.
The only nvcc warning, `compiler-bindir` redefinition, comes from the conda `NVCC_PREPEND_FLAGS`. Later verified on a GPU: committed as 9e297954, all IVF-Flat tests pass (see `pr.md` for the timings).
