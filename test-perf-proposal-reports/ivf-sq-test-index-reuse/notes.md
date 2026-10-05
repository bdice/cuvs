# ivf-sq-test-index-reuse: notes

Test-only change to `NEIGHBORS_ANN_IVF_SQ_TEST`. The only file changed is `cpp/tests/neighbors/ann_ivf_sq.cuh`.
The test sources `cpp/tests/neighbors/ann_ivf_sq/test_{float,half}_int64_t.cu` are unchanged.
This implements ideas D, C and J from `NEIGHBORS_ANN_IVF_SQ_TEST.md` (T1 in `../PROFILING_SUMMARY.md`).

## What changed (line numbers are in the new file)

| where | change |
|---|---|
| `ann_ivf_sq.cuh:65-87` `testAll` | Gets the case's indices from `get_indices()` and runs the same four checks on them, in the same order: Search, Serialize, Filter, Extend. |
| `:95-133` | New `index_key` (`num_db_vecs, dim, nlist, metric, host_dataset`) and `built_indices` (`built`, `loaded`, `extended`, a copy of the creating `raft::resources`, and the device bytes). There is one static LRU cache per fixture instantiation (`index_cache_`), bounded by `kIndexCacheBytes` = 128 MiB. `TearDownTestSuite` clears it. |
| `:150-169` `checkSerialize` | Takes `(built, loaded)` instead of doing the round trip itself. Same `ASSERT_EQ`s (size, dim, n_lists) and the same search comparison (eps 0.001, recall 1.0). |
| `:171-185` `checkExtend` | Takes the extended index. Same search and `eval_neighbours` against the naive reference (eps 0.1, `nprobe/nlist` recall). |
| `:328-336` `make_index_params` | Factored out of `build_index`. The parameters are unchanged. |
| `:338-346` `device_bytes` | Sums centers, list data and list index capacities, for the cache budget. |
| `:348-374` `get_indices` | On a hit, moves the entry to the MRU position and returns it. On a miss, it builds the entry, evicts LRU entries until the new one fits, and inserts it. An entry larger than the budget is used for the current case only and is not cached. |
| `:376-409` `build_indices` | On a miss: (1) `build_index(true)`, as before; (2) serialize to a tmp file and deserialize, as before; (3) **D:** construct an empty `index(handle, params, dim)`, which is the constructor `build()` uses. `raft::copy` the `centers`, `sq_vmin` and `sq_delta` from (1) into it, then call the same `extend_index` as before, which allocates and computes the center norms. This replaces `build_index(false)`, which re-ran the identical k-means + SQ training (same data, same `RngState{137}`). |
| `:567` `inputs` | **J:** deleted the exact duplicate `{1000, 10000, 16, 10, 40, 1024, L2Expanded}` (old `:449`, identical to old `:409`). |

## What is reused

* **Within a case (D):** the extend check reuses the quantizer (centers, `sq_vmin`, `sq_delta`) trained by the `add_data_on_build=true` build.
  Before, it trained an identical one with `build(add_data_on_build=false)`. `build(true)`'s internal extend does not modify these
  arrays. `build(false)` returns exactly an index constructed with the same constructor plus those three arrays (center norms
  unallocated), so the extend path that is verified is the same.
* **Across cases (C):** all three indices (`built`, `loaded`, `extended`) are reused by later cases with the same `index_key`. Those cases differ
  only in `num_queries`, `k` and `nprobe`. The database depends only on `(num_db_vecs, dim)`, because `SetUp` draws it first from
  `RngState(1234)`. Each fixture instantiation (`float`, `half`) has its own cache. No cached index is ever mutated: every
  check takes it by `const&`. On a hit, the case still regenerates its data, computes both naive references, and runs all 5 searches
  plus every assertion with its own search parameters. It skips only the build, the serialize/deserialize and the extend.
* **Not reused:** dataset/queries, naive k-NN, all searches and assertions. The `ExtendInPlaceUpdatesListSizeWithinCapacity` TEST is unchanged.

## Case counts

| test | before | after |
|---|---|---|
| `AnnIVFSQTest/AnnIVFSQTestF_float.AnnIVFSQ/*` | 117 | 116 |
| `AnnIVFSQTest/AnnIVFSQTestF_half.AnnIVFSQ/*` | 17 | 17 |
| `AnnIVFSQTest.ExtendInPlaceUpdatesListSizeWithinCapacity` | 1 | 1 |
| **total** | **135** | **134** |

Name change: float `/0`–`/52` are unchanged; `/53` was the duplicate; old `/54`–`/116` are now `/53`–`/115`. The half names are unchanged.
No in-repo CI script or filter references these indices.

Work per run (default order, no filter). The hit counts come from simulating the cache with estimated entry sizes; the real device bytes are slightly different:

| | before | after |
|---|---|---|
| distinct index keys (float + half) | 99 (84 + 15) | 99 |
| cache hits | n/a | ≈ 30 (28 float + 2 half). An unbounded cache would give 34 |
| `ivf_sq::build` calls (k-means + SQ trainings) in parameterized cases | 268 | ≈ 103 |
| serialize + deserialize round trips | 134 | ≈ 103 |
| `extend` calls (incl. inside build) | 268 | ≈ 206 |

The 4 float misses that an unbounded cache would hit are the dim=16 keys first seen in the dimension block (`/18`–`/21`). The 128 MiB budget evicts them
while the dim 31–256 cases run, before the k / nprobe blocks reuse them. Entries for dim ≥ 2048 (≈ 200–560 MiB) exceed the budget and are never
cached; they also have no repeated keys.

## Expected effect

Baseline from the analysis: 59.4 s gtest time under nsys, 45 s ctest wall time. k-means accounts for 24.0 s and the serialize round trips for 24.1 s.

* D: one training per miss instead of two.
* C: no training, round trip or extends for ≈ 30 cases, ≈ 26 of them at `nlist = 1024` (≈ 0.1 s k-means + ≈ 0.2 s round trip + 2 × 17 ms extend each).
* J: −1 case (it would be a cache hit anyway).

Estimate: ≈ 165 fewer trainings (≈ 15 s), 31 fewer round trips (≈ 5–6 s) and ≈ 62 fewer extends (≈ 1 s). That is **≈ 20 s of 59.4 s nsys gtest time (≈ 35 %), ≈ 15 s of
the 45 s ctest wall time**. Launch- and sync-bound phases are inflated under nsys, so the real saving may be somewhat smaller.

## Risks

* **Shared state across cases.** If a build, serialization or extend is broken, every case sharing the key fails, not only the first. The root cause
  is still reported by the first case of the key. Exceptions during `build_indices` are not cached, so the next case retries.
* **Coverage delta (minor).** The parameterized cases no longer call `build(add_data_on_build=false)`. That branch only skips the trailing
  `extend_inplace` (`ivf_sq_build.cuh:531-533`); the training before it is identical and still runs on every miss. The branch remains
  covered for float/device by `ExtendInPlaceUpdatesListSizeWithinCapacity`. The half and host-input variants of `build(false)` are no longer executed.
  Also, the extend check now uses the same quantizer object as the build check instead of a second, nominally identical training.
* **Peak GPU memory** rises by at most the cache budget (128 MiB) plus one index. All three indices of a case now live for the whole case;
  before, `loaded` and the extended index lived only during their own check. For dim 4096 that is ≈ +140 MiB on top of a ≈ 0.6 GiB case.
* **Hit rate depends on order.** `--gtest_shuffle` and `--gtest_filter` lower it but do not affect correctness. `TearDownTestSuite` frees everything at
  the end of each suite (and each `--gtest_repeat` iteration), so no device memory is held at static destruction.
* **Builds are not bit-reproducible** (static `i_primes` in balanced k-means). Cases that share a key now see the same index instead of
  independent rebuilds, so a borderline recall would fail consistently for all of them instead of randomly. The thresholds are unchanged.
* **Key maintenance.** If an input that affects the database or `index_params` is added to `AnnIvfSqInputs`, it must also be added to `index_key`.
  The struct comment says so.

## How to verify

1. Build and run: `ctest -R NEIGHBORS_ANN_IVF_SQ_TEST`, or run the binary. Expect 134 tests, all passing (before: 135).
2. Isolation: `--gtest_filter='*AnnIVFSQTestF_float*/57'` (a single case, cache miss path) and `--gtest_shuffle --gtest_repeat=2` both pass.
3. Work counted with nsys (`-t cuda,nvtx`):
   * `fused_column_minmax_kernel` launches once per `ivf_sq::build`. Expect 269 before (268 + the TEST) and ≈ 104 after.
   * kvikio `FileHandle` ranges / `serialize` calls: 134 → ≈ 103.
4. Timing: compare the `ctest` wall time and the per-suite gtest times before and after (placeholders in `pr.md`).
5. Compile check: both TUs compile with the build's flags (`-Werror`, `-Wall`, `-Werror=all-warnings`) for `-arch=sm_89`.
