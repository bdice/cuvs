### IVF list (de)serialization does 2 blocking transfers and 2 stream syncs per list

**Problem**

IVF-Flat, IVF-SQ and IVF-PQ save their inverted lists with a loop over `ivf::serialize_list` and load them with a loop over
`ivf::deserialize_list` (`cpp/src/neighbors/ivf_list.cuh`). For files, each non-empty list costs:

* **Write** (`kvikio_ofstream`):
  * 2 × the numpy header is staged, then `sync_stream`, then a flush of the staged header with a blocking kvikio `pwrite`, then a
    blocking kvikio device `pwrite` of the payload (`kvikio_serialize.hpp:78-79`, `file_io.cpp:295-312`).
  * Small payloads go through kvikio's bounce buffer (`cuMemcpy` + sync + `pwrite`). Larger ones go through `cuFileWrite` on the kvikio
    thread pool while the caller waits.
* **Read** (`kvikio_file_reader`):
  * The size scalar is parsed, the list is allocated, and then twice: the header is parsed, the stream is synced, and a blocking kvikio
    `pread` goes into device memory. Each `pread` is followed by a `seekg` that drops the 8 KiB stream buffer.

Every list of an index with `n_lists = 1024` therefore pays ≈ 4 blocking I/O calls and 4 stream syncs. nsys profiles of the test suite
(RTX 6000 Ada) show:

| | serialize + deserialize | per list |
|---|---|---|
| `NEIGHBORS_ANN_IVF_FLAT_TEST` (390 cases) | 98.3 s (27 % of the executable) | ≈ 120 µs at dim 16, ≈ 880 µs at dim 2048 |
| `NEIGHBORS_ANN_IVF_SQ_TEST` (134 cases) | 24.1 s (40 %) | ≈ 155 µs per non-empty list (91 µs write + 65 µs read) |
| `NEIGHBORS_ANN_IVF_PQ_TEST` | ≈ 3.6–4 s per serialize group | same path, not costed |

Users pay the same per-list overhead whenever they save or load an IVF index, whether or not GPUDirect Storage is available. Small lists
are latency-bound, not bandwidth-bound.

**Proposed fix**

Batch the lists while keeping the on-disk format byte-identical:

* **Write.** Copy the payloads of consecutive lists into one pinned host buffer (≤ 64 MiB) with async D2H copies and one sync per batch.
  Then write each record (size scalar, data header, data, indices header, indices) in order as plain stream writes, which
  `kvikio_ofstream` coalesces into 32 MiB `pwrite`s.
* **Read.** Use `list_sizes` (already loaded before the lists) to plan batches. Read each batch with one `pread` into a pinned buffer and
  verify every record's header bytes. Then allocate the lists and issue async H2D copies, with one sync per batch. If a record does not
  match the expected bytes, fall back to the existing per-list parser for the rest of the file.
* **Large lists.** Lists with a payload above a threshold (e.g. 4 MiB) keep the current direct device ↔ file path, so GDS still serves
  large lists.

Not in scope: the per-file cuFile handle register/deregister (≈ 45 ms per file pair in the profiles). It is triggered by the
centers' device writes and the reader's eager GDS handle, not by the lists.
