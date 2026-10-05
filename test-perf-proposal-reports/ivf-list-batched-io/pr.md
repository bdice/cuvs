### Batch IVF list (de)serialization I/O

Before this PR, IVF-Flat, IVF-SQ and IVF-PQ wrote and read every inverted list with 2 blocking kvikio transfers, 2 small header writes
and 2 stream syncs. This PR moves the list loops into two helpers in `cpp/src/neighbors/ivf_list.cuh`, and the three index serializers
call them:

* **`ivf::detail::serialize_lists`.** It stages the payloads of consecutive lists in one pinned buffer (≤ 64 MiB) with async D2H copies
  and syncs once per batch. It then writes the records in order as plain stream writes, which `kvikio_ofstream` turns into a few 32 MiB
  `pwrite`s.
* **`ivf::detail::deserialize_lists`.** For files, it plans batches from the already-loaded `list_sizes` and reads each batch with one
  `pread`. It checks that each record's headers are exactly what the writer produces, then allocates the lists and issues async H2D
  copies, with one sync per batch. If anything does not match, it hands the remaining lists to the existing `deserialize_list`, so
  inconsistent or foreign files are accepted or rejected exactly as before. The `std::istream` overload keeps the per-list loop.
* **Large lists.** Lists with more than 4 MiB of payload still go through `serialize_list` / `deserialize_list`, so GDS keeps serving
  large lists.

The file format is unchanged, byte for byte:

* The record headers come from the same raft/cuVS numpy writers, with the same shapes.
* The payloads are the same device byte ranges.
* The record order is unchanged.

A host-side check compared the new records with both old writer paths, for IVF-Flat (f32/f16/i8/u8), IVF-SQ and IVF-PQ (flat and
interleaved). It also confirmed that the old parsers read them back.

There are no public API changes. Memory use is bounded by one pinned buffer of at most 64 MiB per call. Not addressed here: the
per-file cuFile handle register/deregister, which is triggered by the centers, not the lists.

## Testing

* The full `ctest` suite passes.
* **Format compatibility.** For IVF-Flat, IVF-PQ and IVF-SQ indexes in three configurations each (64 lists over 5,000
  rows; trained on 5,000 rows and extended with 50, so most lists are empty; and 2 lists of 20,000 rows, i.e. above
  the 4 MiB per-list threshold), an index saved by the old library and loaded and saved again by the new one is
  byte-identical, and vice versa. That is 17 of the 18 comparisons. The 18th could not run because the old library
  crashes when saving a *loaded* IVF-PQ index with empty lists. That is a pre-existing bug: `ivf_pq::serialize`
  dereferences the null list pointers that `deserialize` leaves for empty lists. It also happens on `main`, and is
  written up separately. In that case the old library still loads files written by the new one (same `n_lists` and
  size). The new `serialize_lists` writes a null list as size 0, so the new library fixes the crash.

## Measurements

Single-process wall time of each test executable on an RTX 6000 Ada (48 GB) with a 36-core host, otherwise idle (no ctest parallelism, no MPS). Old and new builds were run alternately, 2 or more repetitions each with the order reversed between repetitions; mean (min–max).

The IVF changes on this branch were measured as a chain: each executable was run against the builds before and after each of them (cumulatively), alternating, 2 repetitions per build. For this PR, "before" is the build just before this change and "after" adds only this change. Only `libcuvs.so` differs.

| executable | tests before | tests after | before | after | change |
|---|---|---|---|---|---|
| `NEIGHBORS_ANN_IVF_FLAT_TEST` | 390 | 390 | 92.0 s (91.1–92.8) | 82.8 s (71.7–94.0) | -9.9% (noise; see below) |
| `NEIGHBORS_ANN_IVF_PQ_TEST` | 1065 | 1065 | 158.6 s (158.4–158.7) | 158.6 s (157.9–159.4) | +0.0% |
| `NEIGHBORS_ANN_IVF_SQ_TEST` | 134 | 134 | 20.1 s (20.1–20.2) | 14.9 s (14.9–14.9) | -26.2% |
| `NEIGHBORS_MG_TEST` | 15 | 15 | 14.3 s (14.3–14.3) | 13.8 s (13.7–13.8) | -3.5% |
| `NEIGHBORS_TIERED_INDEX_TEST` | 72 | 72 | 3.0 s (3.0–3.0) | 3.0 s (3.0–3.0) | -0.3% |
| **total** | | | 288.0 s | 273.1 s | -5.2% |

`NEIGHBORS_ANN_IVF_FLAT_TEST` turned out to be noisy here (single runs between 71 and 96 s for the same build), so it
was re-measured with 4 more alternating repetitions per build: 90.0 s (85.2–92.1) before, 91.4 s (81.6–95.9) after.
That is no measurable change. The saving is in IVF-SQ (−26%) and MG (−4%).

All tests passed in every run.

Closes #`<issue>`
