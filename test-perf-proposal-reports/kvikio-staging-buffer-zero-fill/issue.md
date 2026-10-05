### `kvikio_ofstream` zero-fills a 32 MiB staging buffer on every open (~10 ms per index save)

**Problem**

`cuvs::util::kvikio_ofstream` stages ordinary stream output in a host buffer, 32 MiB by default (`cpp/include/cuvs/util/file_io.hpp:466`).
The buffer is a `std::vector<char>` built in `sbuf`'s constructor (`cpp/src/util/file_io.cpp:265,403`). That value-initializes it: every open
`memset`s 32 MiB, even when the file only gets a few KiB of header and the bulk data goes straight to kvikio (`write_device`, or host blocks ≥
the buffer size).

With glibc, a 32 MiB request is always a fresh `mmap`, because the dynamic mmap threshold is capped at 32 MiB. The `memset` therefore faults in all
8,192 pages, and the pages are unmapped again on close. A CPU-only microbenchmark of the allocation with a 4 KiB payload:

| | per open | minor faults |
|---|---|---|
| `std::vector<char>(32 MiB)` (current) | 10.3–12.0 ms | 8,193 |
| `std::make_unique_for_overwrite<char[]>(32 MiB)` | 0.005 ms | 2 |

This matches the untraced 9.7–10.1 ms host gaps that nsys shows right after each `kvikio::FileHandle` open. Every serialize-to-file path
opens at least one stream, including CAGRA, Vamana, brute force, IVF-*, RaBitQ, ScaNN, multi-GPU, and HNSW disk/spill output. In the C++ test suite this adds up to about 1 minute:

* `NEIGHBORS_ANN_VAMANA_TEST`: ≈ 27 s (~2,600 opens)
* `NEIGHBORS_ANN_CAGRA_*_UINT32_TEST`: ≈ 4–10 s each
* `NEIGHBORS_ANN_BRUTE_FORCE_TEST`: ≈ 0.8 s
* `NEIGHBORS_ANN_HNSW_ACE_*`: ≈ 0.4–0.9 s each
* IVF-SQ and RaBitQ: ≈ 1–1.5 s

Users pay the same ~10 ms for every saved index.

**The zero-fill is not needed.** The buffer is read only through `[pbase(), pptr())` in `flush_buffer()`. `xsputn` and `overflow` write every
one of those bytes before advancing `pptr()`, and kvikio's `pwrite` reads exactly `size` bytes from the pointer it is given.

**Proposed fix**

Allocate the staging buffer without value-initialization, using `std::make_unique_for_overwrite<char[]>(n)` plus a stored size (libcuvs builds as
C++20). Behaviour and file contents are unchanged, and pages that are never used are never faulted.

Possible follow-up: allocate the buffer 4096-aligned. kvikio could then use O_DIRECT for full-buffer flushes without copying them through its own bounce buffer.
