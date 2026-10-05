# hnsw-ace-npartitions-dedupe

Test-only change. It removes duplicate builds from the `AnnHnswAceBuild` TEST_P in
`NEIGHBORS_ANN_HNSW_ACE_{FLOAT,HALF,INT8,UINT8}_UINT32_TEST` and keeps every distinct build.

## What changed

`cpp/tests/neighbors/ann_hnsw_ace.cuh:969-993`, `generate_hnsw_ace_inputs()`:

* The `npartitions` axis of the `raft::util::itertools::product` (`:976`) changes from `{0, 1, 2, 4}` to `{2, 4}`.
  All other axes are unchanged: dim `{64, 128}` × use_disk `{false, true}` × metric `{L2Expanded, InnerProduct}`.
* Two cases are appended (`:985-991`) so the 0/1 → 2 resolution path is still exercised:
  * `npartitions=0`, dim 64, `use_disk=false`, L2. This uses the library defaults for both `npartitions` and `use_disk`.
  * `npartitions=1`, dim 64, `use_disk=true`, InnerProduct. This hits the "adjusted to 2" `RAFT_LOG_WARN`.
* A 3-line comment explains why.

The four `test_*_uint32_t.cu` files are unchanged. Each uses `hnsw_ace_inputs` only for `AnnHnswAceBuild`.
`clang-format` 20.1.8 (`cpp/.clang-format`) makes no changes to the edited file.

## Evidence that npartitions 0, 1 and 2 give the same build

1. **The test passes `npartitions` straight through.** `testHnswAceBuild` reads `ps.npartitions` in one place:
   `ace_params.npartitions = ps.npartitions` (`ann_hnsw_ace.cuh:292`). `hnsw::detail::build` copies it into the CAGRA
   `ace_params` without changing it (`cpp/src/neighbors/detail/hnsw.hpp:2995`).
2. **Only `build_ace` reads it.** Across `cpp/src` and `cpp/include`, the only reads of `ace_params.npartitions` are
   `hnsw.hpp:2995` and `cagra_build.cuh:1339`. In `build_ace`, the raw value is used only in the NVTX range label
   (`cagra_build.cuh:1344-1348`) and in `ace_resolve_partition_count` (`:1364`).
3. **Resolution happens before anything else.** `ace_resolve_partition_count` (`cagra_build.cuh:1131-1141`) returns 2
   for 0, 2 for 1 (after a `RAFT_LOG_WARN`), and `n` otherwise. Every later step uses only the resolved `n_partitions`:
   * `ace_validate_partition_count` (`:1366`);
   * the tiny-partition clamp (`:1368-1379`), where 5000 / 1000 = 5 leaves both 2 and 4 unchanged;
   * `ace_check_use_disk_mode` (`:1400`), whose memory estimates and disk-mode decision are computed from
     `n_partitions`;
   * `ace_validate_disk_mode_partitions` (`:1415`), the memory-limit path that can raise `n_partitions`;
   * partition labelling, `min_partition_size`, and the per-partition build loop.
4. **Memory-limit and disk-spill paths.**
   * `AnnHnswAceBuild` uses `max_host_memory_gb = max_gpu_memory_gb = 0`, the system defaults. Disk mode therefore
     depends only on `use_disk` and on memory estimates computed from the resolved count, which is the same for 0/1/2.
   * `AnnHnswAceMemoryFallbackTest` (`:394-445`, inputs `:996-1012`, `npartitions=2`), `AnnHnswAceLayeredTest`
     (`npartitions=2`), `AnnHnswAceInvalidPartitionTest` (overridden to `n_rows+1` at `:752`) and
     `AnnHnswInmemSpillTest` (no ACE) each have their own input lists. This change does not touch them.
5. **The logs agree.** The profiling runs (`../NEIGHBORS_ANN_HNSW_ACE_FLOAT_UINT32_TEST.md`) logged the
   same partition splits for `npartitions` 0, 1 and 2: 2376/2624 rows for L2 and 2333/2667 for IP. The data is seeded
   (`RngState(1234)` in `SetUp`).

**Remaining differences.** With `npartitions=1` the library logs one extra WARN line, and the NVTX range label shows the
raw value. The kept 0 and 1 cases still cover both.

## Case counts

Case index = `dim*8 + np*4 + disk*2 + metric` for the product (first axis varies slowest), then `/16` (np=0) and
`/17` (np=1).

| TEST_P / TEST (per executable) | before | after |
|---|---|---|
| `AnnHnswAceTest/AnnHnswAceTest_<T>.AnnHnswAceBuild` | 32 | **18** |
| `AnnHnswAceInvalidPartitionTest/...RejectsTooManyPartitions` | 1 | 1 |
| `AnnHnswAceMemoryFallbackTest/...AnnHnswAceMemoryLimitFallback` | 1 | 1 |
| `AnnHnswAceLayeredTest/...AnnHnswAceLayeredBuildDeserializeSearch` | 1 | 1 |
| `AnnHnswInmemSpillTest/...AnnHnswFromCagraInmemSpill` | 2 | 2 |
| plain `TEST`s (float: 4; half/int8/uint8: 2) | 4 / 2 | 4 / 2 |

| executable | before | after |
|---|---|---|
| NEIGHBORS_ANN_HNSW_ACE_FLOAT_UINT32_TEST | 41 | 27 |
| NEIGHBORS_ANN_HNSW_ACE_HALF_UINT32_TEST | 39 | 25 |
| NEIGHBORS_ANN_HNSW_ACE_INT8_UINT32_TEST | 39 | 25 |
| NEIGHBORS_ANN_HNSW_ACE_UINT8_UINT32_TEST | 39 | 25 |
| **total** | **158** | **102** (−56; 128 → 72 `AnnHnswAceBuild` ACE builds) |

Coverage of `AnnHnswAceBuild` per `npartitions` value:

| npartitions | before | after |
|---|---|---|
| 0 | 8 (full product) | 1 (dim 64, in-memory, L2) |
| 1 | 8 (full product) | 1 (dim 64, disk, IP) |
| 2 | 8 (full product) | 8 (full product) |
| 4 | 8 (full product) | 8 (full product) |

## Expected effect

Each executable runs 14 fewer ACE build+search cases, about 44% of its largest group. These estimates use
the per-executable profiles (`../NEIGHBORS_ANN_HNSW_ACE_*_UINT32_TEST.md`, idea 1) and their ctest/profile
factors:

| executable | ctest -j8 before | est. saving |
|---|---|---|
| FLOAT | 11.8 s | ~3.5 s (~30%) |
| HALF | 15.8 s | ~4.5 s (~29%) |
| INT8 | 11.0 s | ~2.9 s (~26%) |
| UINT8 | 10.9 s | ~2.9 s (~27%) |

Total ≈ 13.8 s of ctest time. Nothing was run here, so these are estimates. In INT8, `npartitions=0` cases looked slower
only because case `/2` absorbs `CUFileInit` and case `/0` absorbs warm-up. Those one-time costs move to whichever case
runs first and do not go away.

## Risks

* **Renumbered parameter indices.** Cases `/18`–`/31` disappear, and `/0`–`/17` now map to different parameters. For
  example, old `/0` was np=0, dim 64, in-memory, L2; new `/0` is np=2 with the same config. `ci/`, `cpp/tests/CMakeLists.txt`, `conda/` and `.github/` have no gtest filters that reference
  these indices. Out-of-tree filters or dashboards keyed by index would need updating.
* **Narrower coverage if `npartitions=0` ever gains real auto-selection.** `cpp/include/cuvs/neighbors/cagra.hpp:177-179` documents 0 as
  "automatically derived based on available host and GPU memory", but the implementation returns 2. If that is
  implemented, only one `AnnHnswAceBuild` case (dim 64, in-memory, L2) would exercise it. The comment in the generator
  points to `ace_resolve_partition_count` so a future change there can widen the axis again. The doc/implementation
  mismatch is out of scope here.
* **No change to library code, tolerances, recall thresholds or other fixtures.**

## How to verify

1. Build the four targets. Then for each executable:
   `NEIGHBORS_ANN_HNSW_ACE_<T>_UINT32_TEST --gtest_list_tests`. Expect 27 (float) or 25 (others) tests, and
   `AnnHnswAceBuild/0`–`/17`. The `GetParam()` comments should show `npartitions=2` or `4` for `/0`–`/15`,
   `npartitions=0` for `/16` and `npartitions=1` for `/17`.
2. Run each executable. All tests should pass. In the log for `/16` and `/17`, check:
   * `ACE build: start rows=5000 dim=64 partitions=2`;
   * for `/17`, `ACE: Requested 1 partition; adjusted to 2 before applying partitioning heuristics`.
3. Optional equivalence check against the old binary. Compare the `ACE build: start ... partitions=` lines and the
   per-partition sizes from old `/0` (np=0, dim 64, in-memory, L2) with old `/8` (np=2, same config), and old `/7`
   (np=1, dim 64, disk, IP) with old `/11` (np=2, same config). They should be identical.
4. Timing: `ctest -R NEIGHBORS_ANN_HNSW_ACE --output-on-failure -j8` before and after, or each executable on its own.
   Compare against the estimates above.

## Compile check

All four translation units (`cpp/tests/neighbors/ann_hnsw_ace/test_{float,half,int8_t,uint8_t}_uint32_t.cu`) compile
with exit code 0. The flags are the ones from `cpp/build/latest/compile_commands.json`, including
`-Werror=all-warnings` and `-Xcompiler=-Wall,-Werror`, with the arch flags reduced to `-arch=sm_89` and the output in
`$TMPDIR`. Each TU takes about 57 s. The only output is an nvcc driver notice, `incompatible redefinition for option
'compiler-bindir'`. It comes from the conda environment's `NVCC_PREPEND_FLAGS=-ccbin=...`, not from the source.
