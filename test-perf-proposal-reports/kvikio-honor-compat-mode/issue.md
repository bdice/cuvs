### `KVIKIO_COMPAT_MODE=ON` is ignored for device I/O, so every process that saves or loads an index pays ~1 s of cuFile initialization

**Problem**

cuVS opens files for device-memory I/O through `cuvs::util::detail::open_kvikio_file_for_device_io` (`cpp/src/util/kvikio_io.hpp:54-72`).
That function always constructs the `kvikio::FileHandle` with `kvikio::CompatMode::OFF`, and falls back to KvikIO's default mode only if
that throws. As a result:

* **`KVIKIO_COMPAT_MODE=ON` and `kvikio::defaults::set_compat_mode(CompatMode::ON)` have no effect** on these files. cuFile is still loaded,
  every file is registered with `cuFileHandleRegister`, and the first registration in a process initializes the cuFile driver
  (`CUFileInit` in nsys: **0.93–1.15 s per process**, including a ~0.3 s `system()` call).
* **This includes systems where GDS cannot work.** In our devcontainer there is no `nvidia-fs`, `/run/udev` or `cufile.json`. KvikIO's own
  `AUTO` mode would pick POSIX there. cuFile initializes anyway, logs "running in compatible mode", and does POSIX I/O itself
  (`cufile_posix_read/write`). It also adds 5–95 ms of handle register/deregister per file, plus one log line per device I/O in
  `cufile.log`.

The affected paths are every file-based `deserialize` (`kvikio_file_reader`), every serialize with device data
(`kvikio_ofstream::write_device`), `read_large_file` / `write_large_file` with device buffers, and the CAGRA-ACE partition reads. Other
kvikio handles in cuVS already honour the setting.

In the C++ test suite, 14 profiled executables pay `CUFileInit` once per process: brute force, CAGRA ×4, HNSW-ACE ×4, IVF-Flat/PQ/SQ/RaBitQ
and Vamana. UTIL_TEST pays it too. That is ≈ 15 s of test time, and none of it can be avoided with KvikIO's documented switch.

**Reproducer (no GPU needed)**

1. Call `open_kvikio_file_for_device_io` with `KVIKIO_COMPAT_MODE=ON`.
2. `LD_PRELOAD` a stand-in `libcufile.so.0` that logs calls.
3. The log shows that `cuFileHandleRegister` is called and the handle was requested with `CompatMode::OFF`.

**Proposed fix**

* If `kvikio::defaults::compat_mode() == CompatMode::ON`, open the file with `CompatMode::ON` directly and never touch cuFile.
  `AUTO` / `OFF` keep today's behaviour: prefer cuFile, warn once and fall back.
* Run the C++ gtests in CI with `KVIKIO_COMPAT_MODE=ON`. CI runners are not expected to have GDS, so cuFile only runs its POSIX
  fallback there. Keep the cuFile path covered by rerunning UTIL_TEST once in the default mode.

**Related, not part of the fix**

* `AUTO` also bypasses KvikIO's heuristics: no `/run/udev`, WSL, or libcufile missing. Deferring to
  `kvikio::defaults::is_compat_mode_preferred()` would give the same saving in such containers without any variable. It changes default
  behaviour, so it should be discussed separately.
* If libcufile.so.0 cannot be loaded, the forced `OFF` currently aborts the process (`std::terminate`) instead of falling back, because
  KvikIO's `CUFileHandleWrapper::register_handle` is `noexcept`. With the fix, `KVIKIO_COMPAT_MODE=ON` is a working escape hatch for that case.
