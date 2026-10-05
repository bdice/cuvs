# fast-test-verification-helpers

Test-only change to the host-side verification helpers in `cpp/tests/neighbors/ann_utils.cuh`. `calc_recall` (both
overloads) and `check_unique_indices` return exactly the same values and messages as before, for any input. Only the
algorithms changed:

* `calc_recall` was O(rows·k²) and ran two quadratic passes. Long rows are now sorted once and binary-searched,
  O(rows·k log k), in a single pass.
* `check_unique_indices` built a `std::set` per row. It now sorts the row in a reused buffer.

No signature, return type, test case, or threshold changed, and no caller was edited.

Background: T5(a) in `../PROFILING_SUMMARY.md`; `NEIGHBORS_ANN_IVF_RABITQ_TEST.md` idea A, `NEIGHBORS_ANN_NN_DESCENT_TEST.md`
idea D, `NEIGHBORS_ANN_IVF_PQ_TEST.md` idea H, `NEIGHBORS_ANN_IVF_FLAT_TEST.md` idea H.

## What changed

All line numbers refer to `cpp/tests/neighbors/ann_utils.cuh` after the change. The file is formatted with
clang-format 20.1.8 (repo `.clang-format`; it was already clean).

* **Includes** (`:22-33`). Adds `<algorithm>`, `<cmath>`, `<set>`, `<tuple>`, `<type_traits>` and `<vector>`. The
  old code used `std::set` without including `<set>`.
* **`recall_max_scan_cols = 64`** (`:139-143`). Rows of up to 64 expected neighbors are scanned directly. Longer rows
  are sorted. This only affects speed. Scanning a short row is 2.5–3× faster than sorting it at the recall levels the
  tests reach (measured below), and the scan does strictly less work than the old code.
* **`contains_index(row, cols, sorted_row, idx)`** (`:145-153`). Uses `std::binary_search` on the sorted row, or
  `std::find` on the row. Both test `idx` for equality with any expected index, as the old inner loop did.
* **`calc_recall(expected_idx, actual_idx, rows, cols)`** (`:155-185`, index only). Per row, the expected indices are
  copied into a reused buffer and sorted (only if `cols > 64`). Then each actual index is counted if
  `contains_index` finds it. Same `{recall, match_count, total_count}` tuple.
* **`check_unique_indices`** (`:187-228`). Per row, it copies the row into a reused buffer, sorts it, and counts
  adjacent equal pairs whose value is not `std::numeric_limits<T>::max()`. That count equals the old per-row
  duplicate count, Σ(occurrences − 1) over non-sentinel values. If the running total stays within `max_duplicates`,
  it moves on. Otherwise, the row that crosses the limit is replayed in its original order with the old `std::set`
  logic. The replay returns the same `AssertionFailure` message ("Duplicated index … at k … for query …! ") at the same
  `(i, k)`. The unused `max_count` variable is gone.
* **`sorted_distance_lookup<DistT>`** (`:260-263`). True for `float` and `double`. Other distance types (for example
  `half`, or integers, for which `CompareApprox` behaves very differently) always use the scan.
* **`has_approx_equal_distance(row, cols, sorted_row, dist, eps, eq_compare)`** (`:265-307`). Returns whether
  `CompareApprox<DistT>(eps)(e, dist)` holds for any expected distance `e` of the row.
  * Fast path: `float`/`double`, a sorted row, and `0 ≤ (DistT)eps ≤ 0.5`. The row holds the non-NaN distances in
    ascending order. A non-finite `dist` returns false. Otherwise the function computes a window
    `[dist − r, dist + r]` with `r = eps·max(1, |dist|) / (1 − 2⁻¹⁰ − eps)` (in double). It then evaluates the
    unchanged `eq_compare` on every sorted value in the window, closest first, and stops at the first match.
  * Otherwise it evaluates `eq_compare` on every value of the row, as the old loop did.
* **`calc_recall(expected_idx, actual_idx, expected_dist, actual_dist, rows, cols, eps)`** (`:309-370`). One pass.
  Per row, the indices and (for float/double) the non-NaN distances are sorted into reused buffers (only if
  `cols > 64`). For each actual neighbor:
  * if `contains_index` finds it, both `match_count` and `index_match_count` go up;
  * else if `has_approx_equal_distance` finds a distance, `match_count` goes up.

  It returns the same 4-tuple `{recall, index-only recall, match_count, total_count}`. The index-only recall is
  still returned, but it now costs nothing extra: it is the index test that the distance-aware match needs anyway.
  No caller reads it: `eval_neighbours` (`:386`) and `ann_cagra.cuh:1250` only use elements 0, 2 and 3. Keeping it
  avoids touching the callers.
* Unchanged: `idx_dist_pair` (still used by `knn_utils.cuh`), `eval_recall`, `eval_neighbours`, and `eval_distances`.
  `eval_distances`' host loop is already linear.

## Why the results are identical

* **Index matches.** The old loops counted an actual neighbor if *any* expected entry of its row satisfied the
  predicate (`break` on the first). The result does not depend on order, so a sorted copy with `binary_search` (or
  `find`) gives the same answer for every actual neighbor. Duplicate ids in either row count exactly as before. The
  expected side is an existence test, and every actual entry is counted on its own.
* **Distance matches.** `idx_dist_pair::operator==` is `idx == a.idx || CompareApprox(eps)(exp_dist, act_dist)`.
  "Any j with (index equal or distance close)" is the same as "(any j with index equal) or (any j with distance
  close)", so the new code tests the index first and the distance only on a miss. That is also why
  `index_match_count` comes for free.
* **The distance window is a strict superset of the matches.** `CompareApprox<T>(eps)(e, a)` computes
  `diff = |e − a|` and `m = max(|e|, |a|)` in `T`, and accepts if `diff ≤ eps`, or if `diff > eps` and
  `diff / m ≤ eps`. Write δ = |e − a| and let u be the unit roundoff of `T`, with e and a finite:
  * If `diff ≤ eps`, then δ ≤ eps/(1 − u).
  * Otherwise `fl(diff/m) ≤ eps`, so δ ≤ eps·m/(1 − u)². The quotient is never subnormal: for e ≠ a,
    `diff/m ≥ 2⁻²⁵` (float) or `2⁻⁵⁴` (double).
  * With m ≤ |a| + δ this gives δ ≤ eps·max(1, |a|) / ((1 − u)² − eps).
  * Since (1 − u)² > 1 − 2⁻¹⁰ for float and double, `r` is larger than this bound. The double rounding of `r` and of
    `a ± r` cannot shrink the window below it: the bound has ≥ 2⁻¹¹ relative slack, and rounding `a ± r` is monotone
    while `e` is exactly representable in double.
  * Every candidate in the window is checked with the original comparator, so the window only has to contain all
    matches.
* **Special values.**
  * NaN expected distances never match (`diff` is NaN), so dropping them before sorting changes nothing (and keeps
    `std::sort`'s ordering valid).
  * A NaN or ±inf actual distance never matches for a finite `eps`: `diff` is NaN, or `inf/inf` = NaN.
  * ±0 compare equal in the sort and the search.
  * Any `eps` outside `[0, 0.5]` after the cast to `DistT` (negative, NaN, ≥ 0.5, inf) uses the full scan. So do
    `half` and integer distance types.

## Verification (standalone, not committed)

`$TMPDIR/ftvh/check_helpers.cu` includes the working-tree `ann_utils.cuh`, plus the HEAD version of the helpers
(lines 119–316, extracted verbatim with `git show` into `namespace old_impl`). It was compiled with the include and
define flags of the RaBitQ test TU (`-O3`, `-arch=sm_89`, logging compiled out, linked against the build's
`libgtest.a`) and run on the CPU only.

* **Randomized comparison.** 882,112 checks, **0 mismatches**. Each check compares the old and new outputs bit for
  bit: the `calc_recall` tuples (doubles by `memcmp`), and the `AssertionResult` bool plus message for
  `check_unique_indices` (with `max_duplicates` ∈ {0, 1, 2, 7, 10⁶}), `eval_recall` and `eval_neighbours`.
  * Type combinations: (`uint32_t`, `int64_t`, `int`) × (`float`, `double`).
  * Shapes: 0×5, 5×0, 1×1, 7×1, 5×2, 6×3, 4×8, 10×32, 20×64, 4×65, 3×130, 3×257 and 2×1024, which exercise both the
    scan and the sorted path. There are also 16 cases at k = 16384.
  * 20 `eps` values: 0, 1e-45, 1e-7, 1e-4, 0.001, 0.0032, 0.006, 0.01, 0.1, 0.25, 0.4999, 0.5, 0.5000001, 0.75, 1, 2,
    1e30, inf, −0.001 and NaN.
  * Index ids:
    * a range of 1, 2, k/2+1, k+1, 2k+3 or 10⁹, to force duplicates on both sides;
    * a 0–30 % share of `max<T>()` sentinels;
    * distinct or random expected ids;
    * a random hit rate.
  * Distances:
    * continuous, sorted or unsorted;
    * heavy ties with both signs (multiples of 0.5);
    * a 10⁻³⁰–10³⁰ range with both signs;
    * special values (±0, ±inf, NaN, max, lowest, min-normal, ±denorm_min);
    * values around the subnormal range.
  * Actual distances are often placed at the `CompareApprox` acceptance boundary of a random expected value: `e(1±eps)`,
    `e/(1±eps)`, `e±eps`, `−e`, `e(1+2eps)`, each ±0–4 ulps.
* **Sensitivity check.** The test catches deliberately broken variants:
  * dropping the `1/(1 − 2⁻¹⁰ − eps)` factor of the window: 812 mismatches;
  * dropping `max(1, ·)`: 1,510 mismatches;
  * not ignoring the sentinel in the sorted duplicate count: 327 mismatches.

### Standalone speedup

Thread CPU time, best of runs. The data imitates search output: distinct ids, sorted distances, and the given share
of true neighbors. Measured on a shared, loaded machine, so the old absolute times are higher than in the nsys
analyses (7.7 s for the RaBitQ case there). The ratios are what matter.

| shape (rows × k, recall) | `calc_recall` old → new | `check_unique_indices` old → new |
|---|---|---|
| IVF-RaBitQ 64 × 16384, 0.733, eps 0.0032 | 16.1 s → 0.129 s (**125×**) | 0.206 → 0.043 s (4.8×) |
| IVF-PQ 1024 × 2048, 0.9 | 3.51 s → 0.192 s (18×) | 0.289 → 0.063 s (4.6×) |
| IVF-PQ 1024 × 1023, 0.9 | 0.875 s → 0.090 s (10×) | 0.133 → 0.029 s (4.6×) |
| NN-descent 4000 × 64, 0.97 | 16.1 ms → 5.1 ms (3.2×) | 20.3 → 4.2 ms (4.8×) |
| NN-descent 4000 × 32, 0.97 | 5.0 ms → 1.9 ms (2.6×) | 8.0 → 1.7 ms (4.7×) |

Scan vs. sort, new code only (`$TMPDIR/ftvh/xover.cu`; ms for 256 k neighbors):

| k | recall 0.97: sort / scan | recall 0.7 | recall 0.3 |
|---|---|---|---|
| 32 | 10.0 / 3.4 | 10.4 / 5.9 | 10.5 / 8.9 |
| 64 | 12.3 / 4.7 | 12.3 / 9.8 | 12.4 / 17.0 |
| 128 | 14.2 / 7.1 | 14.0 / 18.1 | 14.7 / 34.2 |

That is why the threshold is 64. Above it, sorting wins except at very high recall. At 64, scanning wins down to
recall ≈ 0.6. At recall 0.3 it costs ≈ 1.4× the sorted path, which is still less than the old code.

### Expected effect on the tests (from the analyses' attributions)

| executable | helper time before | expected after |
|---|---|---|
| IVF_RABITQ | `calc_recall` ≈ 7.7 s + `check_unique_indices` ≈ 0.2 s per k = 16384 case, × 5 = 40.5 s of 58 s | ≈ 0.4 s, so ≈ 39–40 s saved (~70 %) |
| NN_DESCENT | `check_unique_indices` 18.0 s + `calc_recall` 8.0 s of 103 s (nsys) | ≈ 6–7 s, so ≈ 19 s n ≈ 14 s real saved |
| IVF_PQ | host recall evaluation 27.7 s n (21.3 s of it k ≥ 1023) | ≈ 2–3 s, so ≈ 25 s n ≈ 15 s real saved |
| IVF_FLAT | ≈ 11 s n of `calc_recall` gaps (1 M / 100 k-query cases) | ≈ 8 s real saved |

The total is ≈ 75–80 ctest seconds, in line with T5(a)'s ≈ 80 c. Other executables that call `eval_neighbours` with
k ≥ 64 gain a little.

## Compile checks

Each TU was compiled with its own `compile_commands.json` command: `-o` redirected to `$TMPDIR`, a single
`-arch=sm_89`, `-t=1`, `nice -n 19`, one at a time, keeping `-Werror` and `-Werror=all-warnings`. Results:

All 12 compiled with exit code 0 and no diagnostics. The only output line was nvcc's driver notice
`incompatible redefinition for option 'compiler-bindir'`, which comes from the environment's `NVCC_PREPEND_FLAGS`
and is not a compiler warning.

* **C++ tests:**
  * `ann_ivf_rabitq/test_float_int64_t.cu`
  * `ann_nn_descent/test_float_uint32_t.cu`
  * `ann_ivf_pq/test_float_int64_t.cu`
  * `ann_cagra/test_float_uint32_t.cu` (the bloom-filter `calc_recall` caller)
  * `ann_cagra/test_bbq_uint32_t.cu` (index-only `calc_recall`)
  * `all_neighbors/test_float.cu` (`eval_recall`)
* **C tests:** all six that include the header: `ann_ivf_pq_c.cu`, `ann_ivf_flat_c.cu`, `ann_ivf_sq_c.cu`,
  `brute_force_c.cu`, `all_neighbors_c.cu` and `ann_mg_c.cu`.

Every instantiation in these tests uses `float` distances, so the sorted-lookup path applies everywhere.

## Affected executables

These 33 CMake targets compile a TU that includes `ann_utils.cuh` directly or indirectly (computed from
`compile_commands.json` plus the transitive `#include` closure; no affected TU is missing from the compile database).
The header's line numbers shift for all of them, so their binaries can change even where the changed helpers are not
called.

* **Material speedup expected:** `NEIGHBORS_ANN_IVF_RABITQ_TEST`, `NEIGHBORS_ANN_NN_DESCENT_TEST`,
  `NEIGHBORS_ANN_IVF_PQ_TEST`, `NEIGHBORS_ANN_IVF_FLAT_TEST`.
* **Other C++ test targets:**
  * `NEIGHBORS_ALL_NEIGHBORS_TEST`, `NEIGHBORS_ANN_BRUTE_FORCE_TEST`, `NEIGHBORS_ANN_IVF_FLAT_UDF_TEST`,
    `NEIGHBORS_ANN_IVF_SQ_TEST`, `NEIGHBORS_ANN_SCANN_TEST`, `NEIGHBORS_ANN_VAMANA_TEST`;
  * CAGRA: `NEIGHBORS_ANN_CAGRA_BBQ_UINT32_TEST`, `NEIGHBORS_ANN_CAGRA_FILTER_UDF_TEST`,
    `NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST`, `NEIGHBORS_ANN_CAGRA_HALF_UINT32_TEST`,
    `NEIGHBORS_ANN_CAGRA_INT8_UINT32_TEST`, `NEIGHBORS_ANN_CAGRA_UINT8_UINT32_TEST`,
    `NEIGHBORS_ANN_CAGRA_MERGE_TEST`, `NEIGHBORS_ANN_CAGRA_TEST_BUGS`;
  * HNSW ACE: `NEIGHBORS_ANN_HNSW_ACE_FLOAT_UINT32_TEST`, `NEIGHBORS_ANN_HNSW_ACE_HALF_UINT32_TEST`,
    `NEIGHBORS_ANN_HNSW_ACE_INT8_UINT32_TEST`, `NEIGHBORS_ANN_HNSW_ACE_UINT8_UINT32_TEST`;
  * `NEIGHBORS_DYNAMIC_BATCHING_TEST`, `NEIGHBORS_HNSW_TEST`, `NEIGHBORS_MG_TEST`, `NEIGHBORS_TEST`,
    `NEIGHBORS_TIERED_INDEX_TEST`.
* **C API test targets:** `ALL_NEIGHBORS_C_TEST`, `BRUTEFORCE_C_TEST`, `IVF_FLAT_C_TEST`, `IVF_PQ_C_TEST`,
  `IVF_SQ_C_TEST`, `MG_C_TEST`.

## Risks

* **Semantics.** The risk is low. The index path is an order-independent existence test. The distance path rests on
  the bound above, and every candidate is still decided by the original `CompareApprox`. The 882 k-check randomized
  comparison, which targets the acceptance boundary at ulp level, found no difference, and the deliberately broken
  windows were caught. Distance types other than float/double, and unusual `eps`, keep the old full scan.
* **Failure messages** are unchanged. The `check_unique_indices` failure text comes from replaying the offending row
  in order. The `eval_*` log lines print the same values.
* **Host floating-point mode.** The bound assumes IEEE round-to-nearest without `-ffast-math`, which matches the
  host flags of the test TUs (`-O3`, no fast-math). Under fast-math, the old comparator itself would be unreliable for
  NaN/inf.
* **Memory.** Three reused buffers of `k` elements per call.
* **Scope.** Only `ann_utils.cuh` changes. Other quadratic helpers outside it are not touched, for example
  `knn_utils.cuh`'s `devArrMatchKnnPair`, which is already linear per row after the optional sort.
