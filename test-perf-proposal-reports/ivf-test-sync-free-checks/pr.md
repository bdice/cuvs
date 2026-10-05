### Make the IVF-Flat and IVF-PQ test checks sync-free per list

Three checks in the IVF test fixtures made a blocking device-to-host round trip per list or per element. This PR
keeps their assertions and moves the round trips out of the loops:

* **IVF-Flat packer check** (`testPacker`).
  * Before: per list, a `thrust::reduce` returned to the host, two `devArrMatch` calls copied and synced, and five
    buffers were allocated.
  * Now: a small `count_if_async` kernel counts the masked elements, the packed-data mismatches and the
    unpacked-data mismatches into a per-list device counter matrix. The counters are read back once after the loop.
  * The host then asserts the same conditions per list, in the same order: masked count `== list_size * dim`, no
    mismatches.
  * Gather and unpack buffers are allocated once.
* **IVF-Flat adaptive-centre check.** The per-list means are computed with the same `copy_selected` and `mean` calls
  into one `n_lists × dim` matrix. They are compared with the index centres after a single copy, using the same
  `CompareApprox<float>(0.001)`.
* **IVF-PQ `compare_vectors_l2`.**
  * Before: it read a managed-memory `device_mdarray` element by element, which is an 8-byte copy and a sync per
    read, and it paid a `cudaMallocManaged`/`cudaFree` per call.
  * Now: it copies the distances to a `std::vector` once. The per-row `ASSERT_LE` and its message are unchanged.

What stays the same:

* The data, ranges, comparators (`==`, `CompareApprox(0.001)`) and tolerances are unchanged. A case fails exactly
  when it failed before.
* Masked-out elements were previously zeroed on both sides before the comparison. They are now skipped, which is
  equivalent.

What differs:

* Packer failure messages report the list and the number of mismatching elements, not the first mismatching value.

No test cases, parameters or library code change.

## Testing

* `NEIGHBORS_ANN_IVF_FLAT_TEST` and `NEIGHBORS_ANN_IVF_PQ_TEST` pass with the same case counts (390 and 1,065), as
  does the full `ctest` suite.

## Measurements

Single-process wall time of each test executable on an RTX 6000 Ada (48 GB) with a 36-core host, otherwise idle (no ctest parallelism, no MPS). Old and new builds were run alternately, 2 or more repetitions each with the order reversed between repetitions; mean (min–max).

The IVF changes on this branch were measured as a chain: each executable was run against the builds before and after each of them (cumulatively), alternating, 2 repetitions per build. For this PR, "before" is the build just before this change and "after" adds only this change. Only the test binaries differ for these executables (the library change in between only touches Vamana).

| executable | tests before | tests after | before | after | change |
|---|---|---|---|---|---|
| `NEIGHBORS_ANN_IVF_FLAT_TEST` | 390 | 390 | 137.7 s (135.5–139.9) | 112.5 s (112.2–112.7) | -18.3% |
| `NEIGHBORS_ANN_IVF_PQ_TEST` | 1065 | 1065 | 166.8 s (166.3–167.4) | 158.2 s (158.1–158.2) | -5.2% |
| **total** | | | 304.5 s | 270.6 s | -11.1% |

All tests passed in every run.

Closes #`<issue>`
