### Don't zero-fill the `kvikio_ofstream` staging buffer

`kvikio_ofstream` allocated its staging buffer (32 MiB by default) as a `std::vector<char>`, so every open `memset` the whole buffer. With glibc,
that means a fresh `mmap` plus 8,192 page faults, about 10 ms per index save, even when the file is only a few KiB of header followed by direct
device writes.

This PR allocates the buffer with `std::make_unique_for_overwrite<char[]>` and keeps its size in a member. The only change is in
`cpp/src/util/file_io.cpp`, and the public API is unchanged.

* **Same output.** Only `[pbase(), pptr())` is ever handed to kvikio, and `xsputn` / `overflow` write those bytes first. A differential test of the
  old and new `sbuf` produced byte-identical files, including with the new buffer pre-filled with garbage under ASan/UBSan.
* **Same errors.** Open and allocation failures still raise `Cannot open file ... for writing`.
* **Never slower.** When the whole buffer is used, both versions fault the same pages.

CPU microbenchmark of a 32 MiB allocation with a 4 KiB payload: **10.3–12.0 ms → 0.005 ms** (8,193 → 2 minor faults).

## Testing

* `UTIL_TEST` (including the `FileIO.KvikioOfstream*` tests) and the full `ctest` suite pass.

## Measurements

Single-process wall time of each test executable on an RTX 6000 Ada (48 GB) with a 36-core host, otherwise idle (no ctest parallelism, no MPS). Old and new builds were run alternately, 2 or more repetitions each with the order reversed between repetitions; mean (min–max).

The library changes on this branch were measured as a chain: each executable was run against six builds of `libcuvs.so` (before any of them, then after each one, cumulatively), alternating, 2 repetitions per build. For this PR, "before" is the build just before this change and "after" adds only this change; the test binaries are identical. Executables affected only by this change were measured separately in the same way.

| executable | tests before | tests after | before | after | change |
|---|---|---|---|---|---|
| `NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST` | 1417 | 1417 | 88.5 s (87.1–89.9) | 84.2 s (83.4–84.9) | -4.9% |
| `NEIGHBORS_ANN_CAGRA_HALF_UINT32_TEST` | 883 | 883 | 51.5 s (50.6–52.4) | 47.9 s (46.7–49.0) | -7.0% |
| `NEIGHBORS_ANN_CAGRA_INT8_UINT32_TEST` | 943 | 943 | 74.5 s (74.4–74.6) | 71.6 s (71.3–71.9) | -3.8% |
| `NEIGHBORS_ANN_CAGRA_UINT8_UINT32_TEST` | 943 | 943 | 79.3 s (78.5–80.1) | 74.3 s (73.3–75.4) | -6.2% |
| `NEIGHBORS_ANN_HNSW_ACE_FLOAT_UINT32_TEST` | 27 | 27 | 7.8 s (7.8–7.9) | 7.6 s (7.3–7.9) | -3.3% |
| `NEIGHBORS_ANN_HNSW_ACE_HALF_UINT32_TEST` | 25 | 25 | 10.5 s (10.4–10.6) | 10.2 s (10.1–10.4) | -2.8% |
| `NEIGHBORS_ANN_HNSW_ACE_INT8_UINT32_TEST` | 25 | 25 | 8.6 s (8.3–8.9) | 7.4 s (7.4–7.5) | -13.4% |
| `NEIGHBORS_ANN_HNSW_ACE_UINT8_UINT32_TEST` | 25 | 25 | 7.6 s (7.1–8.1) | 7.6 s (7.5–7.7) | -0.7% |
| `NEIGHBORS_ANN_CAGRA_BBQ_UINT32_TEST` | 135 | 135 | 3.8 s (3.7–3.9) | 3.9 s (3.8–4.0) | +2.4% |
| `NEIGHBORS_ANN_VAMANA_TEST` | 1260 | 1260 | 316.3 s (315.9–316.7) | 283.8 s (283.7–283.8) | -10.3% |
| `NEIGHBORS_ANN_BRUTE_FORCE_TEST` | 78 | 78 | 5.1 s (5.1–5.1) | 4.0 s (4.0–4.1) | -20.2% |
| `NEIGHBORS_ANN_IVF_SQ_TEST` | 134 | 134 | 23.4 s (23.1–23.7) | 22.3 s (22.1–22.4) | -5.0% |
| `NEIGHBORS_ANN_IVF_RABITQ_TEST` | 230 | 230 | 11.8 s (11.7–11.8) | 10.4 s (10.3–10.4) | -12.1% |
| `UTIL_TEST` | 23 | 23 | 1.4 s (1.4–1.4) | 1.4 s (1.3–1.4) | -2.5% |
| **total** | | | 690.2 s | 636.5 s | -7.8% |

All tests passed in every run.

Closes #`<issue>`
