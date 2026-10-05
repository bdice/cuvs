### Honour `KVIKIO_COMPAT_MODE=ON` when opening files for device I/O

`open_kvikio_file_for_device_io` always opened files with `kvikio::CompatMode::OFF`, so `KVIKIO_COMPAT_MODE=ON` was ignored for device
I/O. cuFile was loaded and every file was registered with it, and the first registration in each process initialized the cuFile driver:
about 1 s, even on systems without GPUDirect Storage, where cuFile only falls back to POSIX.

This PR:

* **`cpp/src/util/kvikio_io.hpp`:** if `kvikio::defaults::compat_mode()` is `CompatMode::ON` (from `KVIKIO_COMPAT_MODE=ON` or
  `set_compat_mode`), the file is opened with `CompatMode::ON` directly and cuFile is never touched. `AUTO` (the default) and `OFF` are
  unchanged. No API, ABI or file-format changes.
* **`ci/test_cpp.sh`:** the C++ gtests run with `KVIKIO_COMPAT_MODE=ON`; a job can override it. The CI runners are not expected to have
  GDS, so this only drops cuFile's own POSIX fallback. To keep the cuFile open path covered, shard 1 reruns `UTIL_TEST` with
  `KVIKIO_COMPAT_MODE=AUTO` (about 2.5 s).

Coverage trade-off: most gtests no longer exercise the cuFile layer below KvikIO. Everything above it is unchanged: cuVS serialization,
KvikIO's `FileHandle`, and its POSIX device path.

## Testing

* The full `ctest` suite passes with `KVIKIO_COMPAT_MODE=ON`, and `UTIL_TEST` passes with `KVIKIO_COMPAT_MODE=AUTO`
  (the CI rerun added here).
* Checked without a GPU, using a stand-in `libcufile.so.0`:
  * before, with `ON`: `cuFileHandleRegister` was still called.
  * after, with `ON`: the handle is `CompatMode::ON` and libcufile is never loaded.
  * `AUTO` / `OFF`: unchanged.

## Measurements

Single-process wall time of each test executable on an RTX 6000 Ada (48 GB) with a 36-core host, otherwise idle (no ctest parallelism, no MPS). Old and new builds were run alternately, 3 or more repetitions each with the order reversed between repetitions; mean (min–max).

Both builds ran with `KVIKIO_COMPAT_MODE=ON` (as `ci/test_cpp.sh` now sets it), 3 repetitions each; the old library ignores the variable for device I/O and initializes cuFile. Only `libcuvs.so` differs.

| executable | tests before | tests after | before | after | change |
|---|---|---|---|---|---|
| `UTIL_TEST` | 23 | 23 | 1.3 s (1.3–1.4) | 0.5 s (0.5–0.5) | -65.3% |
| `NEIGHBORS_ANN_BRUTE_FORCE_TEST` | 78 | 78 | 4.3 s (4.2–4.3) | 3.7 s (3.6–3.7) | -13.6% |
| `NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST` | 1417 | 1417 | 24.3 s (24.1–24.5) | 23.3 s (23.2–23.3) | -4.1% |
| `NEIGHBORS_ANN_CAGRA_HALF_UINT32_TEST` | 883 | 883 | 14.4 s (14.2–14.5) | 13.3 s (13.2–13.3) | -7.5% |
| `NEIGHBORS_ANN_HNSW_ACE_FLOAT_UINT32_TEST` | 27 | 27 | 4.1 s (4.1–4.1) | 3.1 s (3.0–3.2) | -23.3% |
| `NEIGHBORS_ANN_HNSW_ACE_HALF_UINT32_TEST` | 25 | 25 | 5.8 s (5.7–5.8) | 4.9 s (4.9–5.0) | -14.1% |
| `NEIGHBORS_ANN_HNSW_ACE_INT8_UINT32_TEST` | 25 | 25 | 3.9 s (3.9–3.9) | 2.9 s (2.9–3.0) | -25.1% |
| `NEIGHBORS_ANN_HNSW_ACE_UINT8_UINT32_TEST` | 25 | 25 | 3.9 s (3.9–3.9) | 2.9 s (2.9–3.0) | -25.3% |
| `NEIGHBORS_ANN_IVF_SQ_TEST` | 134 | 134 | 14.9 s (14.7–15.1) | 14.0 s (13.8–14.3) | -5.4% |
| `NEIGHBORS_ANN_IVF_RABITQ_TEST` | 230 | 230 | 10.2 s (10.1–10.3) | 9.6 s (9.0–10.4) | -5.7% |
| **total** | | | 87.0 s | 78.3 s | -10.0% |

All tests passed in every run.

Closes #`<issue>`
