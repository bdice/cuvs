### [TEST] Host-side recall and uniqueness checks in `ann_utils.cuh` are quadratic and dominate several ANN tests

**Problem**

Most ANN tests verify their results with `eval_neighbours` / `eval_recall` from `cpp/tests/neighbors/ann_utils.cuh`.
These run two single-threaded host helpers:

* `calc_recall` is O(n_queries·k²). For every actual neighbor, it scans the whole expected row. The distance-aware
  overload does this **twice**: once for the index-or-distance match, and once more for an index-only recall. Both
  callers throw that second result away (`eval_neighbours` and the CAGRA bloom-filter check only read elements 0, 2
  and 3 of the tuple).
* `check_unique_indices` builds a `std::set` for every row.

Profiles of the test executables (nsys on an RTX 6000 Ada; the GPU is idle during these phases) show:

* `NEIGHBORS_ANN_IVF_RABITQ_TEST`: the `k = 16384` case (64 queries) spends ≈ 7.7 s in `calc_recall` and ≈ 0.2 s
  in `check_unique_indices`, in each of its 5 build paths. That is **40.5 s of the executable's 58 s (70 %)**.
* `NEIGHBORS_ANN_NN_DESCENT_TEST`: `check_unique_indices` takes 18.0 s and `calc_recall` 8.0 s, **26 % of 103 s**.
* `NEIGHBORS_ANN_IVF_PQ_TEST`: host recall evaluation takes 27.7 s, 21.3 s of it in the k ≥ 1023 cases (≈ 1.9 s of
  idle GPU per k = 2048 case).
* `NEIGHBORS_ANN_IVF_FLAT_TEST`: ≈ 0.6 s idle gaps per large-query case, ≈ 11 s in total.

**Proposal**

This is a test-only change to `ann_utils.cuh`, with no change to what is verified:

* `calc_recall`: sort each expected row (indices, and non-NaN distances) once into reused buffers, then look up each
  actual neighbor by binary search. This takes O(n_queries·k log k), in a single pass.
  * The index-only recall is still returned. It is a by-product of the index lookup, so it costs nothing.
  * Distance matches are found by searching a window around the actual distance that provably contains every value
    `CompareApprox<DistT>(eps)` can accept. Each candidate is still decided by the original comparator, so the counts
    are bit-identical, including for ties, duplicate ids, sentinels, ±0, inf and NaN.
  * Short rows (k ≤ 64) are scanned directly in the same single pass, which is faster than sorting them.
  * Other distance types and unusual `eps` values keep the full scan.
* `check_unique_indices`: sort each row into a reused buffer and count adjacent duplicates, ignoring the
  `max<T>()` sentinel. Only the row that exceeds `max_duplicates` is replayed in order with the old logic, so the
  failure message is unchanged.

**Expected impact**

In standalone measurements of the helpers:

* the RaBitQ shape gets **≈ 125×** faster;
* the IVF-PQ k = 2048 / 1023 shapes get 18× / 10× faster;
* the NN-descent shapes get ≈ 3× faster;
* `check_unique_indices` gets ≈ 4.5–5× faster.

Using the profiles' attribution, that is ≈ 75–80 s of ctest time:

* RaBitQ ≈ 39 s (~70 % of the executable);
* NN-descent ≈ 14 s;
* IVF-PQ ≈ 15 s;
* IVF-Flat ≈ 8 s.

Every other test that calls these helpers with k ≥ 64 also gets a little faster. No test case, threshold or signature
changes.
