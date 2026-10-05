# Batch the list-pointer uploads in IVF `recompute_internal_state`

`ivf::detail::recompute_internal_state` used to upload the per-list data and index pointers with one 8-byte
`cudaMemcpyAsync` per list and per array: 2·`n_lists` pageable H2D copies on every extend, deserialize,
`extend_list`/`erase_list`, clone and device refine. This PR gathers the pointers into two host vectors and
uploads each array with a single `raft::copy` on the same stream.

## Changes (`cpp/src/neighbors/ivf_common.cuh`)

* The loop fills `std::vector`s of the device arrays' value types, with the same
  `list ? list->data_ptr() : nullptr` and `list ? list->indices_ptr() : nullptr` values as before.
* Two `raft::copy(dst, src, n_lists, stream)` calls replace the 2·`n_lists` single-element copies.
* The size sort, the D2H copy, the stream sync and `accum_sorted_sizes` are unchanged.

This covers IVF-Flat, IVF-PQ and IVF-SQ, and everything built on them (MG, tiered index, CAGRA's IVF-PQ graph
build, device `refine`).

## Why results don't change

* The device arrays get the same values, computed from the same `index.lists()`.
* The copies are enqueued on the same stream, before the same sort kernel.
* The function already calls `sync_stream` before returning, so the function-scope host buffers outlive the
  copies.

## Testing

* The full `ctest` suite passes, including all IVF-Flat, IVF-PQ and IVF-SQ executables, MG, tiered index,
  dynamic batching and refine (`NEIGHBORS_TEST`).

## Measurements

Single-process wall time of each test executable on an RTX 6000 Ada (48 GB) with a 36-core host, otherwise idle (no ctest parallelism, no MPS). Old and new builds were run alternately, 2 or more repetitions each with the order reversed between repetitions; mean (min–max).

The IVF changes on this branch were measured as a chain: each executable was run against the builds before and after each of them (cumulatively), alternating, 2 repetitions per build. For this PR, "before" is the build just before this change and "after" adds only this change. Only `libcuvs.so` differs.

| executable | tests before | tests after | before | after | change |
|---|---|---|---|---|---|
| `NEIGHBORS_ANN_IVF_FLAT_TEST` | 390 | 390 | 142.3 s (140.3–144.3) | 137.7 s (135.5–139.9) | -3.3% |
| `NEIGHBORS_ANN_IVF_PQ_TEST` | 1065 | 1065 | 167.7 s (167.6–167.8) | 166.8 s (166.3–167.4) | -0.5% |
| `NEIGHBORS_ANN_IVF_SQ_TEST` | 134 | 134 | 21.5 s (21.1–21.9) | 21.0 s (20.7–21.3) | -2.2% |
| `NEIGHBORS_MG_TEST` | 15 | 15 | 14.4 s (14.3–14.4) | 14.3 s (14.3–14.3) | -0.5% |
| `NEIGHBORS_TIERED_INDEX_TEST` | 72 | 72 | 3.1 s (3.1–3.1) | 3.0 s (3.0–3.0) | -3.1% |
| **total** | | | 349.0 s | 342.8 s | -1.8% |

All tests passed in every run.

Closes #`<issue>`
