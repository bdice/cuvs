# kvikio-honor-compat-mode: notes

Library change to `libcuvs.so` plus a CI change. `change.patch` touches two files:

* `cpp/src/util/kvikio_io.hpp`: `open_kvikio_file_for_device_io` honours `KVIKIO_COMPAT_MODE=ON`.
* `ci/test_cpp.sh`: the C++ gtests run with `KVIKIO_COMPAT_MODE=ON`, and UTIL_TEST is rerun once in the default mode.

This implements T4(c) of `../PROFILING_SUMMARY.md`: brute-force idea 2, HNSW-ACE idea 3, IVF-SQ idea K.

## Mechanism: why cuFile is initialized even when it is not wanted

KvikIO picks the I/O backend per `kvikio::FileHandle`. The constructor's last argument is `CompatMode compat_mode = defaults::compat_mode()`.
`defaults::compat_mode()` comes from `KVIKIO_COMPAT_MODE` (`ON` / `OFF` / `AUTO`, case-insensitive; default `AUTO`) or from
`defaults::set_compat_mode()`. `CompatModeManager` (`kvikio/cpp/src/compat_mode_manager.cpp:35-87`) then works like this:

* `ON`: POSIX I/O, with O_DIRECT where it is possible. libcufile is never loaded.
* `AUTO`: resolves once per process (`defaults::infer_compat_mode_if_auto`, `defaults.cpp:200-209`). It becomes `OFF` only if
  `is_cufile_available()` is true, which needs three things: libcufile.so.0 loads, `/run/udev` is a directory, and the host is not WSL.
  Otherwise it becomes `ON`. If an `AUTO` handle fails to open O_DIRECT or to register with cuFile, it falls back to POSIX.
* `OFF`: opens O_DIRECT and calls `cuFileHandleRegister`. The first registration in a process implicitly opens the cuFile driver. That is
  the `CUFileInit` NVTX range in the profiles: 0.93–1.15 s, including a ~0.3 s `system()` call. If the O_DIRECT open or the
  registration fails, the constructor throws.

cuVS does not pass the user's setting through for device I/O. `open_kvikio_file_for_device_io` (`kvikio_io.hpp:54-72`) always constructs
the handle with `CompatMode::OFF` (through the SFINAE helper `open_kvikio_file_compat_off`). Only if that throws does it log a one-time
"GDS is unavailable" warning and reopen with the default mode. So:

* **`KVIKIO_COMPAT_MODE=ON` is ignored for device I/O.** cuFile is still loaded and initialized, and every device-I/O file is registered
  and deregistered.
* **`AUTO`'s heuristics are bypassed too.** This devcontainer has no `/run/udev` (it is a Docker container), no `/dev/nvidia-fs*`, no
  `nvidia_fs` kernel module, no `/etc/cufile.json` and no `CUFILE_ENV_PATH_JSON`. KvikIO's `AUTO` therefore resolves to POSIX here.
  cuVS forces `OFF` anyway, and cuFile then runs in *its own* compatibility mode:
  `gtests/cufile.log` shows `failed to open /proc/driver/nvidia-fs/devcount` and `running in compatible mode` once per process (22×).
  The profiles show `cufile_posix_read/write`. The ~1 s of initialization buys nothing.

Callers that reach this function, all inside `libcuvs.so`:

* `kvikio_file_reader` (`file_io.cpp:213`). Every file-based `deserialize` opens one, even before any device read.
* `kvikio_ofstream::write_device` (`file_io.cpp:308`). It opens the handle lazily, on the first device-memory write.
* `read_large_file` / `write_large_file`, through `open_kvikio_file_for_ace_io`, when the buffer is device memory.
* The CAGRA-ACE partition reads (`cagra_build.cuh:923-957`), also through `open_kvikio_file_for_ace_io`.

All other kvikio handles already use `defaults::compat_mode()` and already honour the variable: `kvikio_ofstream`'s main handle,
host-buffer ACE I/O, and `hnsw.hpp:1660-1662`.

The same mechanism has other costs:

* Per-file cuFile register/deregister: ≈ 5 ms + 15–95 ms per file (IVF-Flat analysis, ≈ 45 ms per serialize case).
* On this host, cuFile logs one `checkIfHostMem` WARN line per device I/O: 165 k lines, 23 MB in `gtests/cufile.log`.

## Change

```cpp
#if CUVS_KVIKIO_HAS_COMPAT_MODE_HEADER
  if (kvikio::defaults::compat_mode() == kvikio::CompatMode::ON) {
    return kvikio::FileHandle(path, flags, kvikio::FileHandle::m644, kvikio::CompatMode::ON);
  }
#endif
  // unchanged: try CompatMode::OFF, warn once and fall back to the default mode on failure
```

* It also adds `#include <kvikio/defaults.hpp>` inside the existing `__has_include(<kvikio/compat_mode.hpp>)` block.
* The check is guarded by the existing `CUVS_KVIKIO_HAS_COMPAT_MODE_HEADER`. `get_kvikio.cmake` pins KvikIO to the same RAPIDS version,
  so the header is always present. With an older KvikIO, the behaviour is unchanged.
* It passes `CompatMode::ON` explicitly instead of re-reading the default, so the handle matches the mode that was checked.
* Public API, ABI and file formats are unchanged. Only the TUs that include `kvikio_io.hpp` are rebuilt: `file_io.cpp` and the
  CAGRA TUs that include `cagra_build.cuh`.

## Behaviour matrix (device-memory opens through `open_kvikio_file_for_device_io`)

| setting | before | after |
|---|---|---|
| unset (= `AUTO`) | `OFF` first: cuFile load + driver init (~1 s/process) + per-file register/deregister. On failure: warn once and reopen with `AUTO`. | **unchanged** |
| `AUTO` | same as unset | **unchanged** |
| `ON` (or `set_compat_mode(ON)`) | **ignored**: same as `AUTO` | `CompatMode::ON`: POSIX, libcufile never loaded |
| `OFF` | `OFF`. On failure, reopen with the default (= `OFF`), which throws again, so the error propagates. | **unchanged** |
| libcufile.so.0 not loadable (any setting) | `std::terminate` (see "Side finding") | `ON`: works. Others: unchanged |

Host-memory opens and the non-device handles listed above are unaffected in every row.

### Verified without a GPU (`probe/`)

`probe.cpp` calls `open_kvikio_file_for_device_io` on a small file and reports the handle's requested and effective mode. It was built
once against the unpatched header and once against the patched one. It ran with `CUDA_VISIBLE_DEVICES=` and an `LD_PRELOAD`ed stand-in
`libcufile.so.0` that logs every cuFile call and fails registration, so neither the real cuFile driver nor the GPU is touched.

| `KVIKIO_COMPAT_MODE` | before | after |
|---|---|---|
| unset / `AUTO` | `cuFileHandleRegister` called, warning, falls back to POSIX | same |
| `ON` | **`cuFileHandleRegister` called**, warning, falls back to POSIX | no cuFile call; requested=`ON`, preferred=`ON`; without the stand-in, libcufile and libcuda are never mapped |
| `OFF` | register called, fallback reopen fails, exception propagates | same |

Both TUs were compile-checked with their exact `compile_commands.json` commands against an overlay copy of `cpp/src`, with `nice -n 19`,
`-o` to `$TMPDIR` and `-arch=sm_89`. `-M` confirmed that the overlay `kvikio_io.hpp` was used. Both are clean under `-Wall -Werror` /
`-Werror=all-warnings`:

* `cpp/src/util/file_io.cpp` (g++, 10 s)
* `cagra_build_inst_data_f_index_u32.cu` (nvcc, 56 s), which includes `cagra_build.cuh`

The CAGRA *serialize* TUs do not include `kvikio_io.hpp`; they reach it through `file_io.cpp`. The header passes clang-format 20.1.8,
and `ci/test_cpp.sh` passes shellcheck 0.11 with no new findings. `git apply --check` passes.

## Should `AUTO` also skip the forced `OFF`? (not in this patch)

A one-line variant would replace the check with `kvikio::defaults::is_compat_mode_preferred()`, which is true for `ON`, and for `AUTO`
when KvikIO itself would choose POSIX.

* **Gain.** Containers without `/run/udev`, including this devcontainer and probably CI, would skip `CUFileInit` without any environment
  variable. It would also fix the libcufile-missing abort for `AUTO`.
* **No gain where AUTO resolves to OFF.** On hosts with libcufile and `/run/udev`, KvikIO's `AUTO` resolves to `OFF` and pays `CUFileInit`
  too, so deferring to `AUTO` saves nothing there. Only `ON` avoids it everywhere.
* **Cost.** It changes default behaviour. The #2257 comment ("Prefer GDS for device transfers") suggests that overriding `AUTO` was
  intentional. A container
  with GDS devices passed through but no `/run/udev` would stop trying GDS. KvikIO's docs say `AUTO` falls back in that case; whether
  GDS can work there at all was not checked.

This is left as a follow-up for the maintainers and the #2257 author. The requested change is limited to the explicit `ON`.

## Side finding: abort when libcufile cannot be loaded (pre-existing, not fixed here)

In this KvikIO version, `CUFileHandleWrapper::register_handle` (`kvikio/cpp/src/file_utils.cpp:97`) is `noexcept`. Under `OFF` it calls
`cuFileAPI::instance()`, which throws when `dlopen("libcufile.so.0")` or a symbol lookup fails. That exception hits the `noexcept`, so the
process calls `std::terminate`, and cuVS's `try/catch` never sees it. With `dlopen` of libcufile forced to fail (`probe/no_cufile.c`), the
unpatched probe aborts for every setting, including `ON`. The patched probe works with `ON`.

Packaged installs ship libcufile (conda `libkvikio` depends on `libcufile-dev`; wheels load it through `cuda.pathfinder`), so this is an
edge case. Examples are source builds against a toolkit without cuFile, or a broken environment. A proper fix belongs in KvikIO
(`register_handle` should not be `noexcept`, or should catch) or in the `AUTO` variant above.

## CI side (`ci/test_cpp.sh`)

* `export KVIKIO_COMPAT_MODE="${KVIKIO_COMPAT_MODE:-ON}"` before the gtests, with a comment saying why. A job can override it, for
  example `AUTO` on a GDS-capable runner.
* In shard 1 only, when the default `ON` is in effect, UTIL_TEST is rerun with `KVIKIO_COMPAT_MODE=AUTO`. That is today's default path:
  forced `OFF`, cuFile registration, `cuFileRead`/`cuFileWrite` or cuFile's own POSIX fallback, and the warn-and-fallback branch. It goes
  through `read_large_file`, `kvikio_ofstream::write_device` and `kvikio_file_reader::read_device`. Cost: UTIL_TEST (~1.4 s) plus one
  `CUFileInit` (~1 s), once per CI run.

### Does CI have GDS?

Unknown from this repository. RAPIDS CI runs `ci/test_cpp.sh` inside the `rapidsai/ci-conda` container through shared-workflows
`conda-cpp-tests.yaml`. Whether `/run/udev`, `/dev/nvidia-fs*` or a `cufile.json` are passed in is not visible here. KvikIO's own CI says
"CI/CD machines don't support running GDS" (`kvikio/ci/test_java.sh:39`). This devcontainer has none of the prerequisites, and cuFile
logs "running in compatible mode". The most likely case is that CI currently exercises cuFile's POSIX fallback, not real GDS.

### Coverage trade-off

* **Lost for most gtests:** the cuFile layer under KvikIO, meaning handle register/deregister, `cuFileRead`/`cuFileWrite` on the KvikIO
  thread pool, and cuFile's internal POSIX fallback. Also cuVS's forced-`OFF` + fallback branch.
* **Kept by the UTIL_TEST rerun:** that layer and that branch, in one process.
* **Kept for all tests:** every cuVS serialization format and code path above KvikIO, KvikIO's `FileHandle` API, and KvikIO's POSIX device
  path (bounce buffers, O_DIRECT for aligned chunks).
* **Not covered either way:** real GPUDirect DMA, because no runner appears to have it.

### Alternatives considered

* ENVIRONMENT on each test in `ConfigureTest` (`cpp/tests/CMakeLists.txt`). This also speeds up local `ctest`, but it hides the
  cuFile path from developers who do have GDS, and it is harder to override.
* A separate full-suite GDS job doubles test cost and needs a GDS runner.

## Affected tests / who pays `CUFileInit`

Confirmed in the nsys profiles (one `CUFileInit` per process, 0.93–1.15 s):

* `NEIGHBORS_ANN_BRUTE_FORCE_TEST`
* `NEIGHBORS_ANN_CAGRA_{FLOAT,HALF,INT8,UINT8}_UINT32_TEST`
* `NEIGHBORS_ANN_HNSW_ACE_{FLOAT,HALF,INT8,UINT8}_UINT32_TEST`
* `NEIGHBORS_ANN_IVF_FLAT_TEST`, `NEIGHBORS_ANN_IVF_PQ_TEST`, `NEIGHBORS_ANN_IVF_SQ_TEST`, `NEIGHBORS_ANN_IVF_RABITQ_TEST`
* `NEIGHBORS_ANN_VAMANA_TEST`

That is 14 executables.

Also expected, by code but not profiled:

* `UTIL_TEST` (`FileIO.DeviceReadRoundTrip`, `KvikioOfstreamMixedHostAndDeviceWrites`, `KvikioFileReaderMixedStreamAndDeviceReads`, ...)
* Possibly `NEIGHBORS_MG_TEST`, `NEIGHBORS_ANN_CAGRA_MERGE_TEST`, `NEIGHBORS_HNSW_TEST`, `NEIGHBORS_TIERED_INDEX_TEST` and
  `NEIGHBORS_ANN_IVF_FLAT_UDF_TEST`, if they serialize to files with device data.

The profiled executables BBQ, FILTER_UDF, TEST_BUGS, NN_DESCENT and SCANN showed no `CUFileInit`.

Expected savings:

* ≈ 1 s per paying process, so ≈ 15 s of summed gtest time per full suite. Wall time under `ctest -j8` gains less, depending on scheduling.
* Plus the per-file register/deregister: ≈ 45 ms per serialize case in IVF-Flat, 15–70 ms per large file in IVF-SQ.
* Plus cuFile's per-I/O logging on hosts like this one.

## How to measure

Only `libcuvs.so` changes; the gtest binaries are identical. The gtests carry `DT_RPATH`, so `LD_LIBRARY_PATH` cannot redirect them. Run
them against the before/after `libcuvs.so` with `LD_PRELOAD=<build>/libcuvs.so`, or swap the library.

1. **Per-process cost, 2 × 2 matrix.** Use {before, after} × {`KVIKIO_COMPAT_MODE` unset, `=ON`}, single process (no ctest
   parallelism), ≥ 3 alternating repetitions, each in a fresh empty working directory (cuFile writes `cufile.log` to the CWD). Use
   `UTIL_TEST`, `NEIGHBORS_ANN_BRUTE_FORCE_TEST` (≈ 5 s), `NEIGHBORS_ANN_IVF_FLAT_TEST` and `NEIGHBORS_ANN_HNSW_ACE_FLOAT_UINT32_TEST`.
   For CAGRA, use `NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST` with a `--gtest_filter` subset that includes serialize cases.
   * Expected: before/unset ≈ before/ON ≈ after/unset, because the variable is ignored today.
   * after/ON should be ≈ 1 s faster per process, plus ≈ 45 ms per IVF serialize case.
   * Also compare the gtest time of the first file-opening case, e.g. brute-force case /0 is 1.14 s today.
2. **Attribution.**
   * With `nsys profile -t nvtx,osrt`, the `CUFileInit` NVTX range should be present in three cells and absent in after/ON.
   * Cheaper: `LD_DEBUG=files <exe> 2>&1 | grep -c 'file=libcufile'` should be 0 for after/ON.
   * On non-GDS hosts like this one: `grep -c 'running in compatible mode' cufile.log` should be 1 per process before, and no
     `cufile.log` at all for after/ON.
3. **Which executables pay.** Run each gtest once (after build, unset) in its own empty directory, then check for `cufile.log` /
   `running in compatible mode` or the `CUFileInit` NVTX range.
4. **Bulk-I/O check.** `ON` replaces cuFile's POSIX fallback with KvikIO's POSIX device path for large transfers. Compare whole-executable
   times, not just the first case, for the I/O-heavy executables: BRUTE_FORCE `dim ≥ 2048`, IVF_FLAT dim 2048, HNSW_ACE spill.
5. **Suite level.** Run `ctest -j8` with the after build, with and without `KVIKIO_COMPAT_MODE=ON`, alternating.

## Risks

* **Default users: none.** Without the variable, the code path is the same; the only addition is one read of
  `kvikio::defaults::compat_mode()`.
* **Users who set `ON`** now get exactly what the variable documents: no cuFile. Anyone who set `ON` but relied on cuVS still using GDS
  would lose GDS. That seems unlikely, because the variable exists to disable cuFile.
* **Bulk throughput under `ON` is unmeasured.** KvikIO's POSIX path (16 MiB bounce buffer, thread pool, O_DIRECT) replaces cuFile's POSIX
  fallback, which the brute-force profile showed as 4 MiB D2H copies + `pwrite`s at ~1 GB/s. It is likely the same or faster, but
  check it with step 4.
* **CI coverage** of the cuFile layer shrinks to one UTIL_TEST run. See the trade-off above.
