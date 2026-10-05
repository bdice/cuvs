### Make the host recall and uniqueness checks in the ANN tests O(k log k)

`calc_recall` in `cpp/tests/neighbors/ann_utils.cuh` was O(n_queries·k²) and ran two quadratic passes. The second
pass computed an index-only recall that no caller reads. `check_unique_indices` built a `std::set` per row. Together
they take 70 % of `NEIGHBORS_ANN_IVF_RABITQ_TEST` (its k = 16384 case) and about a quarter of
`NEIGHBORS_ANN_NN_DESCENT_TEST`, and they leave the GPU idle for seconds in the large-k IVF-PQ / IVF-Flat cases.

This PR changes only the test helpers. Results and messages are identical:

* **`calc_recall`** (both overloads) makes one pass.
  * Rows longer than 64 are sorted once into reused buffers and binary-searched; short rows are scanned.
  * The index-only recall is still returned, now as a by-product of the index lookup.
  * Distance matches are looked up in a window around the actual distance that provably contains every value
    `CompareApprox<DistT>(eps)` can accept. Every candidate in it is still decided by that comparator.
  * Non-float/double distances and `eps` outside `[0, 0.5]` keep the full scan.
* **`check_unique_indices`** sorts each row into a reused buffer and counts adjacent duplicates, still ignoring the
  `max<T>()` sentinel. The row that exceeds `max_duplicates` is replayed in order with the old logic, so the failure
  reports the same index, `k` and query.

No signatures, callers, test cases or thresholds change.

A standalone program compared the old and new helpers on randomized inputs and found 0 differences in 882 k
checks. The inputs covered:

* ties and duplicate ids, `max<T>()` sentinels, ±0, ±inf, NaN and subnormals;
* distances placed within a few ulps of the `CompareApprox` boundary;
* k = 1 to 16384;
* 20 `eps` values;
* `uint32_t`/`int64_t`/`int` × `float`/`double`.

The same harness catches deliberately broken windows. On the IVF-RaBitQ shape (64 × 16384), `calc_recall` goes from
16.1 s to 0.13 s of CPU time.

## Measurements

Single-process wall time of each test executable on an RTX 6000 Ada (48 GB) with a 36-core host, otherwise idle (no ctest parallelism, no MPS). Old and new builds were run alternately, 2 or more repetitions each with the order reversed between repetitions; mean (min–max).

Both builds use the same `libcuvs.so`; only the test binaries differ. Other ANN test executables use these helpers too; only the four with the largest profiled cost were timed.

| executable | tests before | tests after | before | after | change |
|---|---|---|---|---|---|
| `NEIGHBORS_ANN_IVF_RABITQ_TEST` | 230 | 230 | 50.2 s (49.8–50.6) | 11.7 s (11.7–11.7) | -76.6% |
| `NEIGHBORS_ANN_NN_DESCENT_TEST` | 916 | 916 | 75.9 s (75.2–76.7) | 51.6 s (51.4–51.8) | -32.1% |
| `NEIGHBORS_ANN_IVF_FLAT_TEST` | 390 | 390 | 159.7 s (158.0–161.5) | 148.9 s (147.2–150.7) | -6.7% |
| `NEIGHBORS_ANN_IVF_PQ_TEST` | 1065 | 1065 | 279.0 s (278.3–279.6) | 258.4 s (257.2–259.7) | -7.4% |
| **total** | | | 564.8 s | 470.7 s | -16.7% |

All tests passed in every run.

Closes #`<issue>`
