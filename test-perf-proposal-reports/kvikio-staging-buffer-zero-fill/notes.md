# kvikio-staging-buffer-zero-fill: notes

Library change to `libcuvs.so`. Only `cpp/src/util/file_io.cpp` changes (`change.patch`); no header changes.
This implements T4(a) of `../PROFILING_SUMMARY.md`: Vamana idea 4, CAGRA float idea 6, HNSW-ACE idea 4 and brute-force idea 3.

## What changed (line numbers are in the new file)

| where | change |
|---|---|
| `file_io.cpp:17-18` | `#include <memory>` replaces `#include <vector>`. Nothing else in the file uses `std::vector`. |
| `:266-272` `sbuf::sbuf` | The staging buffer is `std::make_unique_for_overwrite<char[]>(buffer_size_)` instead of `std::vector<char>(n)`. The size is still `max(cap, kNumpyDataAlignment)`, and the `pbump` range check and the `setp` are the same. |
| `:345`, `:392` | `buffer_.size()` / `buffer_.data()` become `buffer_size_` / `buffer_.get()`. |
| `:405-411` members | `std::vector<char> buffer_` becomes `size_t buffer_size_; std::unique_ptr<char[]> buffer_;`, with a comment saying why the buffer is left uninitialized. `buffer_size_` is declared before `buffer_` because `buffer_`'s initializer reads it. |

The open order is unchanged: the `FileHandle` is opened first, then the buffer is allocated. The error behaviour is also unchanged: an open or
allocation failure still becomes `RAFT_FAIL("Cannot open file %s for writing: ...")` through the function-try-block, so
`FileIO.KvikioOfstreamOpenFailureIncludesPath` is unaffected. There are no public API or ABI changes, because `sbuf` is private to the `.cpp`.

## Why it costs ~10 ms per open

* `kvikio_ofstream` defaults to a 32 MiB staging buffer (`include/cuvs/util/file_io.hpp:466`), and every index save opens at least one.
* `std::vector<char>(n)` value-initializes the buffer, which means a `memset` of 32 MiB.
* glibc caps its dynamic mmap threshold at 32 MiB (`DEFAULT_MMAP_THRESHOLD_MAX` on 64-bit). A 32 MiB + header request is therefore always a fresh
  `mmap`, and the `memset` faults in all 8,192 4-KiB pages. THP is `madvise` here, so no huge pages are used. Closing the stream `munmap`s the
  buffer, so the next open pays the same cost again.
* With an uninitialized allocation, only the pages that actually stage bytes are faulted. Serializers send large host blocks (≥ buffer size)
  and all device data (`write_device`) straight to kvikio, so for the test indices usually only a few header pages are touched.

## Microbenchmark (CPU only, `nice -n 19`, single thread)

The source is `bench.cpp` in this directory. It was built with the conda g++ 14.4 using libcuvs's `-O3 -std=gnu++20 -march=nocona -mtune=haswell`
and run on the system glibc 2.39. Each iteration allocates the buffer, `memcpy`s the "staged" bytes into it, and frees it. The table shows the median of 40 iterations.
Two runs were taken while the host load average was 33–36.

| capacity | bytes staged | `std::vector<char>(n)` (current) | `make_unique_for_overwrite` (patch) | minor faults/iter (current → patch) |
|---|---|---|---|---|
| 32 MiB | 4 KiB (header-only file) | **10.3 / 12.0 ms** | **0.005 ms** | 8,193 → 2 |
| 32 MiB | 1 MiB | 10.3 / 11.3 ms | 0.52 / 0.34 ms | 8,193 → 257 |
| 32 MiB | 32 MiB (buffer fully used) | 21.7 / 15.3 ms | 15.5 / 15.9 ms | 8,193 → 8,193 |
| 1 MiB (`buffered_ofstream`) | 4 KiB | 0.042 ms | 0.000 ms | 6 → 0 |
| 1 MiB (`buffered_ofstream`) | 1 MiB | 0.10 ms | 0.06 ms | 0 → 0 |

* The ~10–12 ms per open matches the untraced 9.7–10.1 ms gaps after each `FileHandle` open in the Vamana, CAGRA and brute-force profiles.
  That confirms the inferred attribution. The 31–33 ms gaps in the HNSW-ACE profiles are most likely the same work slowed down by
  contention from concurrent threads; those savings may come out smaller.
* When the whole buffer is used, both versions fault every page. The patch is then neutral, within noise; it never adds work.

## Safety: the zero-fill is not needed

* **Only staged bytes are read.** The buffer is read in one place: `flush_buffer()` passes `[pbase(), pptr())` to `kvikio::FileHandle::pwrite`.
  `pptr()` advances in two places, `xsputn` (`memcpy` and then `pbump(n)`) and `overflow` (`*pptr() = ch` and then `pbump(1)`), and both write the bytes first.
  After each flush, `setp` resets the range to empty.
* **No other path reads it.** Large writes (`remaining >= buffer_size_`) bypass the buffer. `write_device` flushes first and then writes the
  caller's pointer. `seekoff` only computes `offset_ + (pptr() - pbase())`.
* **kvikio reads exactly `[buf, buf + size)`.** `posix_host_io` (`kvikio/detail/posix_io.hpp:120-185`) writes the buffered prefix and suffix
  directly. For the O_DIRECT middle it copies `bytes_requested ≤ size` into its bounce buffer. It never reads past `size`.
* **Differential test.** The `sbuf` class was extracted verbatim from the old and the new `file_io.cpp`, with `kvikio::FileHandle` mocked by POSIX `pwrite`.
  * It ran 13 random write sequences: `put`, small, mid-size and ≥ capacity `write`s, `write_device` with host pointers, random `flush`, and `tellp` checked after every op.
  * The capacities were 1 (→ 4096), 4096, 5000, 64 KiB and 32 MiB.
  * Old and new produced byte-identical files that also matched the expected stream, and `bytes_written()` / `tellp()` matched.
  * It was rerun with the new buffer pre-filled with `0xA5` (simulated garbage) under ASan + UBSan, and the output was still identical. So no unwritten
    byte ever reaches the file.
  * Valgrind could not run on this host (stripped `ld.so`).
* **Readers and other helpers:**
  * `kvikio_file_reader` reads through `fd_streambuf`, which already uses `new char[8192]` (uninitialized).
  * `read_large_file` and `write_large_file` have no staging buffer.
  * `buffered_ofstream` (public header, a 1 MiB `std::vector<char>`, used once per HNSW→hnswlib export at `detail/hnsw.hpp:1343`) has
    the same pattern. It costs ≤ 0.04 ms per use, because the memory is recycled heap with no faults, and its pages are filled anyway.
    Changing it would touch a public header and rebuild every TU that includes `file_io.hpp`, so it is left as is.
  * The `std::vector<char>` link buffers in `detail/hnsw.hpp:981,1258,2769,2825` are working buffers that get fully written. They are not per-open staging buffers.

## Build checks

* **Compile.** The exact `compile_commands.json` command for `src/util/file_io.cpp` was used, with the source pointing at the modified copy, `-o` in
  `$TMPDIR`, and `nice -n 19`. It compiled cleanly under `-Wall -Werror` (gnu++20, conda g++ 14.4).
  * One flag was added: `-iquote /home/coder/cuvs/cpp/src/util`.
  * Reason: the `.cpp` includes `"kvikio_io.hpp"` with quotes, which resolves relative to the including file's directory. That directory
    no longer exists for a copy in `$TMPDIR`, so the flag points it at the repo's unchanged copy. No header was modified.
* **Format.** `clang-format` 20.1.8 (the pre-commit version) with `cpp/.clang-format` reports no changes.
* **Patch.** `git -C /home/coder/cuvs apply --check change.patch` passes. `file_io.cpp` is unmodified in the working tree, so the patch also applies to HEAD.
* **C++ level.** `cuvs_objs` is `CXX_STANDARD 20` (`cpp/CMakeLists.txt:1554-1557`). `std::make_unique_for_overwrite` needs libstdc++ ≥ 11.
  If an older toolchain must be supported, `std::unique_ptr<char[]>(new char[n])` is equivalent; that form is already used by `fd_streambuf` in `file_io.hpp`.

## Affected tests (what to measure)

Only `libcuvs.so` changes (one object, then a relink). Every save through a `kvikio_ofstream` gets ~10 ms faster:

| executable | path through `kvikio_ofstream` | expected saving (from the profiles) |
|---|---|---|
| `NEIGHBORS_ANN_VAMANA_TEST` | `vamana::serialize`: dataset, index and sector-aligned files (`vamana_serialize.cuh:45,335,370`); ~2,646 opens | ≈ 27 s (≈ 8 %) |
| `NEIGHBORS_ANN_CAGRA_{FLOAT,HALF,UINT8,INT8}_UINT32_TEST` | `AnnCagraTest::testCagra` → `cagra::serialize(filename)` (`ann_cagra.cuh:610`, `cagra_serialize.cuh:180`) | ≈ 10 / 9.4 / 5.5 / 4.3 s under nsys (≈ 3 s for float after T1) |
| `NEIGHBORS_ANN_HNSW_ACE_{FLOAT,HALF,INT8,UINT8}_UINT32_TEST` | disk-mode `from_cagra` and in-memory spill → `exclusive_hnsw_output_file` (`detail/hnsw.hpp:121,2264,2363`) | ≈ 0.4–0.9 s each |
| `NEIGHBORS_ANN_BRUTE_FORCE_TEST` | `brute_force::serialize(filename)` (`brute_force_serialize.cu:53,64`); 78 opens | ≈ 0.84 s |
| `NEIGHBORS_ANN_IVF_SQ_TEST`, `NEIGHBORS_ANN_IVF_RABITQ_TEST` | `ivf_sq_serialize.cuh:75`, `ivf_gpu.cu:291` | ≈ 1.5 s, ≈ 1.2 s |
| `NEIGHBORS_ANN_IVF_FLAT_TEST`, `NEIGHBORS_ANN_IVF_PQ_TEST` | `ivf_flat_serialize.cuh:85`, `ivf_pq_serialize.cuh:105` | small: ~10 ms × number of serialize cases (not costed) |
| `NEIGHBORS_MG_TEST` | `snmg.cuh:787` plus the per-algorithm file serializers | small (not costed) |
| `UTIL_TEST` | `FileIO.KvikioOfstream*` unit tests of this class | correctness check, not timing |

Not affected:
* `NEIGHBORS_ANN_CAGRA_BBQ_UINT32_TEST` and iterative CAGRA-Q serialize into a `std::stringstream`.
* `NEIGHBORS_ANN_SCANN_TEST` and `NEIGHBORS_ANN_NN_DESCENT_TEST` do not serialize.
* `NEIGHBORS_HNSW_TEST` builds in memory and spills only when host memory is short.

The C API tests (`c/tests/neighbors/ann_cagra_c.cu`, `ann_hnsw_c.cu`, `run_mg_c.c`) also pass through these code paths.

Suggested measurement: run `ctest -R 'NEIGHBORS_ANN_VAMANA_TEST|NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST|NEIGHBORS_ANN_BRUTE_FORCE_TEST'` and `UTIL_TEST`, before and after.

## Risks and limits

* **Risk: low.** The bytes written are identical. Peak RSS of a save goes down, because untouched pages are never faulted.
* **Inferred gaps.** The per-test savings come from profiler gaps that were attributed to the zero-fill by inference. The microbenchmark reproduces
  the per-open cost (10–12 ms) but not the in-process timing; the HNSW-ACE 31–33 ms gaps in particular may not shrink fully.
* **Allocator behaviour.** Allocators other than glibc (jemalloc, tcmalloc) may hand back recycled, already-faulted memory. The fix still
  removes the 32 MiB `memset` (~1–3 ms) there.
* **Not done (possible follow-ups):**
  * *Lazy allocation.* Not needed: an untouched uninitialized allocation costs one `mmap`/`munmap` pair (~5 µs measured).
  * *Page-aligned staging buffer.* glibc returns a page + 16 B pointer, so kvikio copies every full-buffer O_DIRECT flush through its bounce
    buffer (`posix_io.hpp:141-153`; `KVIKIO_AUTO_DIRECT_IO_WRITE` defaults to on). A 4096-aligned buffer would avoid that copy. That changes the I/O path, so it belongs in a separate change.
