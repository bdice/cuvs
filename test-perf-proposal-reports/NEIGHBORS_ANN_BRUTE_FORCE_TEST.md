# NEIGHBORS_ANN_BRUTE_FORCE_TEST: where the time goes and how to make it faster

Source: 2 nsys runs, one per fixture group (`-t cuda,nvtx,osrt`, RTX 6000 Ada). Tests: `ann_brute_force/test_{float,half}.cu`
→ `ann_brute_force.cuh`. This is a **small test** (5 s under `ctest -j8`). Most of that time is not brute force: it is
the serialize → deserialize round trip through kvikio/cuFile that every case does (`ann_brute_force.cuh:108-129`).

## Summary

| metric | value |
|---|---|
| cases / groups | 78 / 2 (`AnnBruteForceTest_float`, `AnnBruteForceTest_half_float`; 39 inputs each, `ann_brute_force.cuh:168-214`) |
| gtest time (nsys) | 6.5 s (3.4 + 3.1). nsys wall time is 8.3 / 7.6 s, mostly nsys launch and export. |
| process start-up | ~0.5 s before the first case (library load). Inside case /0: CUDA context 0.11 s, module load 0.03 s. |
| GPU busy | 0.7 / 0.5 s = **20 % / 16 %** of the CUDA span. 411 kernels per group, 0.38 / 0.33 s of kernel time. |
| slowest cases | case /0: 1.14 / 1.20 s (`CUFileInit`). `{1000, 500000, 128}`: 0.51 / 0.34 s. Seven `dim ≥ 2048` cases: 0.10–0.19 s each. |
| median case | 14 ms, of which 10 ms is a host-only gap in `serialize` |

## Where the time goes (sum of both groups: 6.27 s of per-case time)

Cases are split at the `rngKernel` pair in `SetUp`. The I/O window runs from the first kvikio NVTX range to the last one.

| bucket | s | % | what it is |
|---|---|---|---|
| bulk file I/O | 2.0 | 32 % | Mostly the 16 cases with 41–256 MB datasets: 1.17 s float, 0.74 s half. cuFile has no GDS here and falls back to its own POSIX path (`cufile_posix_write`). Each 4 MiB chunk does a D2H copy into a bounce buffer and then a `pwrite`, about 1 GB/s on one kvikio thread. Reads are ~3× faster. Closing / `cuFileHandleDeregister` on large files takes up to 78 ms. |
| `CUFileInit` | 1.9 | 30 % | 0.93 / 0.96 s, **once per process**, at the first `serialize` (case /0). The main thread spends 0.28 s in `system()`, ~0.1 s in ~58 k `statx`/`openat`/`fopen` calls, 0.1 s in `usleep`, and the rest in untraced CPU. It is forced by `CompatMode::OFF` (`cpp/src/util/kvikio_io.hpp:54-72`). |
| `kvikio_ofstream` zero-fill | 0.84 | 13 % | 78 × 10.1 ms host-only gap right after the `FileHandle` opens: a 32 MiB `std::vector<char>` (`cpp/src/util/file_io.cpp:265,403`). This is 80 % of a small case. |
| kernels | 0.7 | 11 % | `naive_distance_kernel` (test reference) 0.53 s, `fusedL2kNN` 0.12 s, cuBLAS GEMMs < 0.05 s |
| other | 0.8 | 13 % | cuBLAS init in the first InnerProduct case (0.05 / 0.15 s), host-side result checks, allocations, syncs |

**What the ~1 s GPU-idle gaps are.** The single 0.98 s gap at t ≈ 0.6 s (after case /0's `fusedL2kNN`) is `CUFileInit`,
inside the first `brute_force::serialize`. The rest of the 2.6–2.8 s idle time per group is host-side file I/O:
* 39 × 10 ms zero-fills;
* the `pwrite`/`pread` loops and handle close/deregister on the large files.

The GPU work itself is only 0.5–0.7 s per group.

## Ideas (ranked by estimated savings; % of the 6.5 s profiled)

1. **Do the file round trip only on small indexes** *(REDUCES coverage)*
   - Change: guard `ann_brute_force.cuh:108-129` with `num_db_vecs * dim * sizeof(DataT) <= 8 MiB`. This skips the 7 `dim ≥ 2048` cases and `{1000, 500000, 128}` per dtype.
   - Coverage: still kept: every metric/dim search, and multi-chunk (> 4 MiB) serialize through `{10000, 40000, 32}` float (5 MB). Lost: serialize of large files only.
   - Savings: **1.9 s (29 %)**. Effort: trivial. Risk: low.
2. **Skip `CUFileInit` in test runs** *(REDUCES coverage of the cuFile layer only)*
   - Change: in `open_kvikio_file_for_device_io` (`kvikio_io.hpp:54-72`), do not force `CompatMode::OFF` when `kvikio::defaults::compat_mode() == CompatMode::ON`. Then set `KVIKIO_COMPAT_MODE=ON` for gtests (ENVIRONMENT in `ConfigureTest`, `cpp/tests/CMakeLists.txt:22`).
   - Why it is safe: without GDS, cuFile already uses POSIX I/O, so files and the kvikio code above cuFile stay tested. Keep one CI job without the variable.
   - Savings: **~0.95 s per process**: 1.9 s (29 %) here with 2 processes, ~0.95 s (≈ 19 %) of the 5 s ctest run. Every serialize-testing executable gains the same (CAGRA, IVF-*, Vamana).
   - Effort: S. Risk: low–medium (honouring the env var changes user-visible behaviour, arguably for the better).
3. **Do not zero-fill the 32 MiB staging buffer** *(KEEPS coverage)*
   - Same as CAGRA_FLOAT idea 6 and VAMANA idea 4 (`file_io.cpp:265,403`).
   - Savings: **0.84 s (13 %)**, about 0.7 s if idea 1 is applied. Effort: S. Risk: low.
4. **Not worth changing:** the naive reference (0.53 s), data generation, cuBLAS init, and the brute-force kernels.
   Suite level only: `PERCENT 100` (`cpp/tests/CMakeLists.txt:214`) reserves the whole GPU for a test with a ~1.4 GB
   NVML peak (the earlier PERCENT sweep suggested 15).

Ideas 1+2+3 together: ~4.5 s of the 6.5 s profiled. Under ctest, roughly 5 s → ~2 s.

## Uncertainties

* No CPU sampling. Two attributions are therefore inferred:
  - the 10 ms gap is the zero-fill: it follows the `FileHandle` open with no traced calls, the same evidence as in the CAGRA/Vamana docs;
  - what `CUFileInit`'s `system()` call runs is unknown: the export recorded no file paths.
* I/O cost depends on `/tmp` (overlay on local disk here) and the page cache, so CI hosts may differ. OSRT tracing adds per-call overhead to the `pwrite`/`pread` loops.
* Idea 2 assumes kvikio's own POSIX path moves bulk data as fast as cuFile's POSIX fallback. Not measured; no GPU runs or builds were allowed.
* ctest figures are scaled from profiled shares. ctest runs one process (one `CUFileInit`) under `-j8` GPU sharing.
