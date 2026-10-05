> **Update after rebasing onto `upstream/main` (2026-10-05).** #2665 (merged) already caches `CUDA_CACHE_PATH` in CI
> with the shared-workflows cache inputs, which is what this proposal first did (`change_full_cache.patch`). The commit
> on `test-perf-proposals` is now only the follow-up: a per-shard cache key and cache-size logging (`change.patch`).
> The analysis and measurements below still apply.

# ci-jit-cache: cold CUDA JIT cache cost of the C++ tests

## Summary

* **Measured (this branch, RTX 6000 Ada, `ctest -j8`, 59 tests, `KVIKIO_COMPAT_MODE=ON`).** Warm JIT cache: 857.9 / 854.2 s
  wall. Cold (fresh `CUDA_CACHE_PATH`, as in a fresh CI container): 1580.1 s wall (+724 s, +85%). The sum of per-test
  times is 962 s warm and 1658 s cold (+696 s). The cold run wrote 666,368,737 B in 1,337 files, below the 1 GiB
  default `CUDA_CACHE_MAXSIZE`.
* **Mechanism.** rtcx (a dependency) links each JIT-LTO kernel with nvJitLink (`-lto -arch=sm_XX`) into a cubin and
  loads it with `cudaLibraryLoadData`. Within a process, rtcx caches the launcher per fragment set. Across processes,
  nvJitLink caches its own output in the CUDA driver's JIT cache (`CUDA_CACHE_PATH`) at two levels: linked NVVM IR to
  cubin, and PTX to cubin. Warm links take 20–40 ms. Cold links take ≈ 5 s (CAGRA single-CTA), ≈ 1.1 s (multi-CTA)
  and ≈ 0.1 s (multi-kernel).
* **Why CAGRA costs ≈ 140 s per data type.** One executable links ≈ 60 distinct CAGRA search kernels (float: ≈ 19
  single-CTA, ≈ 16 multi-CTA, 8 + 6 multi-partition, ≈ 12 multi-kernel). The key is algorithm × metric × (team size,
  dataset block dim) × filter, plus the VPQ parameters for CAGRA-Q. The ≈ 27 single-CTA links at ≈ 5 s each are about
  80% of the cost. The single-CTA kernel is the expensive one because it compiles every bitonic-sort size behind
  runtime branches. Half is about half as expensive, because its tests never reach the 256/512 dimension buckets.
* **Recommendation: persist the JIT cache across CI jobs.** The fix is in cuVS: `.github/workflows/pr.yaml`,
  `test.yaml` and `ci/test_cpp.sh` (`change.patch`). shared-workflows already has the inputs needed
  (`cache-paths`, `cache-key-*`, `cache-environment`, `cache-read-only`; rapidsai/shared-workflows#628, #633), and cudf
  already uses them for its JIT cache.
  * The tests on main save one archive per shard and configuration. PRs restore it read-only.
  * Expected saving: ≈ 700 s of test time per CUDA/GPU configuration, and ≈ 12 runner-minutes for each of the 4 (of
    6) PR configurations that have a nightly twin. This holds whenever the PR does not change the JIT fragments.
* **Cache entries stay valid across CI builds.** nvcc names internal symbols with
  `_INTERNAL_<crc32(absolute source path)>_<n>_<file>_<crc32(first external symbol)>`, and these names are part of the
  cache key. RAPIDS CI builds libcuvs at a fixed path (`rattler-build --no-build-id --output-dir
  $RAPIDS_CONDA_BLD_OUTPUT_DIR`), so an unchanged fragment produces an identical key in the next build.
* **Library and test changes:** none recommended for CI time. Optional: specialise the single-CTA kernel on its sort
  sizes, which may make its links several times cheaper (unmeasured). For developers, set `CUDA_CACHE_MAXSIZE=4 GiB`;
  the local cache is full at 1 GiB and evicting.

## 1. Mechanism

### Who links, and how

* cuVS only selects fragments; the link and load are in rtcx (`cpp/build/latest/_deps/rtcx-src`, commit `a9f63f8`).
  * A CAGRA search builds a planner (`cpp/src/neighbors/detail/cagra/jit_lto_kernels/cagra_jit_launcher_factory.hpp:37-308`).
    The planner adds static LTO-IR fragments:
    * `setup_workspace` and `compute_distance` for (team size, dataset block dim), or for (pq_len, smem dtype) with VPQ
      (`cagra_planner_base.hpp:44-161`);
    * `dist_op` per metric (`:203-240`) and a normalization (cosine or no-op, `:242-268`);
    * the search kernel (`search_single_cta_planner.hpp:40-105`, `search_multi_cta_planner.hpp:203-207`,
      `search_multi_kernel_planner.hpp:285-302`);
    * the sample filter (`cagra_planner_base.hpp:388-403`).
  * It then calls `get_launcher()`. IVF-PQ (`ivf_pq/detail/jit_lto_kernels/compute_similarity_planner.hpp:15-90`),
    IVF-Flat, IVF-SQ, RaBitQ and the pairwise distances work the same way.
* `rtcx::algorithm_planner::build()` (`_deps/rtcx-src/src/algorithm_planner.cpp:57-110`):
  * calls `nvJitLinkCreate` with `-lto -arch=sm_<cc>` (`:66-77`). No cuVS planner sets `linktime_extra_options`
    (`include/rtcx/algorithm_planner.hpp:53`).
  * calls `nvJitLinkAddData(NVJITLINK_INPUT_ANY)` for each fragment (`src/fragment_entry.cpp:10-16`), then
    `nvJitLinkComplete` (`:86`) and `nvJitLinkGetLinkedCubin` (`:90-96`).
  * loads the result with **`cudaLibraryLoadData` on the cubin** (`:102-104`) and `cudaLibraryGetKernel` (`:107`).
  * The loaded image is SASS, so loading involves no driver PTX JIT.
* The fragments are LTO-IR fatbins embedded in libcuvs. The JIT fragments are compiled with
  `--generate-code=arch=compute_75,code=[lto_75] -rdc=true -fatbin` (`compile_commands.json`, for example
  `ivf_pq_compute_similarity_out_f_lut_f_kernel.cu`).

### What is cached, and where

* **In-process:**
  * Each planner class has a static `launcher_jit_cache` (for example `search_single_cta_planner.hpp:24`). Its key is
    the concatenation of the fragment keys (`algorithm_planner.cpp:24-31, 42-55`).
  * A static fragment's key is `typeid(static_fatbin_fragment_entry<Tag>).name()` (`fragment_entry.hpp:45-48`). A UDF
    fragment's key contains the generated source (`sample_filter_udf.cuh:77-85`).
  * So each distinct fragment set is linked at most once per process.
  * `get_launcher` holds the planner cache's write lock while it links (`:49-53`), so first-time links in one planner
    class are serialized. That does not matter for the tests.
* **Across processes: nvJitLink, through the driver's JIT cache (`CUDA_CACHE_PATH`, default `~/.nv/ComputeCache`;
  `CUDA_CACHE_MAXSIZE`, default 1 GiB; `CUDA_CACHE_DISABLE`).** The evidence comes from parsing the 2,233 entries of
  the local cache.
  * Each entry is a 58-byte header, then the key, then the value. The header is `u32`, then `u64` (key size + 30),
    `u64` value size, `u64` hash, and the stamp `"Sep  1 2026" "08:51:46" "HOST64" "sm_89"`.
  * The stamp string occurs in `libnvJitLink.so.13.4.92` and not in `libcuda.so.590.48.01`. So the entries are
    versioned by the nvJitLink build. A new nvJitLink release misses everything once.
  * 1,233 entries have key = linked **NVVM IR bitcode** (`BC\xC0\xDE`) and value = **cubin**. 1,000 have key = **PTX**
    and value = cubin. Every PTX key is a JIT-LTO kernel (`compute_similarity`, `search_single_cta`, …). None is a
    regular libcuvs kernel, so there is no driver PTX JIT of the library itself.
  * Both kinds are still being written on 10-05 (408 NVVM, 269 PTX), so a cold link writes one or two entries.
  * A hit on the NVVM key skips both the NVVM optimisation and ptxas. That is the 20–40 ms warm cost.
  * The earlier warm-cache profile measured 1.4 s for single-CTA UDF misses, but a fully cold cache gives 5.1 s. A
    hit at the PTX level, which skips ptxas, would explain the difference. This is inference; it was not traced.
* **Not cached across processes:** NVRTC of UDF sources (≈ 75 ms each). cuVS does not use rtcx's own on-disk
  `rtcx::cache_t` (`include/rtcx/rtcx.hpp:575`). The only rtcx symbols cuVS uses are `algorithm_planner`,
  `algorithm_launcher`, `launcher_jit_cache`, `udf_fatbin_fragment` and `nvrtc_compiler`.

### Are entries valid across libcuvs builds? Yes, if the build path is fixed (RAPIDS CI: yes)

* Every NVVM key contains `_INTERNAL_<h1>_<n>_<file>_cu_<h2>` names for each linked fragment. Examples: CCCL CPO
  objects such as `…cuda3std3__45__cpo9iter_swapE`, which survive into the PTX.
* `h1` = CRC32 of the **absolute, symlink-resolved path** of the fragment source. Checked:
  * `crc32("/home/coder/cuvs/cpp/build/conda/cuda-13.3/release/generated_kernels/ivf_pq/compute_similarity/ivf_pq_compute_similarity_out_f_lut_f_kernel.cu")`
    = `062f806f`;
  * the same holds for a CAGRA `compute_distance` fragment.
* `h2` = CRC32 of the TU's first external symbol name. Checked: `crc32("compute_similarity")` = `21f09b20`,
  `crc32("interleaved_scan")` = `d6da037f`, `crc32("apply_filter_kernel")` = `2be5e780`.
* cudafe++ falls back to time and pid (`make_module_id: str1 = %s, str2 = %s, pid = %ld`) only when no external
  symbol exists. Every JIT fragment defines an external kernel or device function.
* Fragments recompiled on 10-05 04:26 carry the same ids as cache entries from 10-01 and 10-02. Example: RaBitQ
  `bitwise_block_sort` keeps (`9f498d4e`, `f84ed908`).
* So, for the same compiler and dependencies, an unchanged fragment produces the same key in a new build **if the
  build directory is the same**.
  * RAPIDS CI builds with `rattler-build --no-build-id --output-dir "$RAPIDS_CONDA_BLD_OUTPUT_DIR"` (gha-tools
    `rapids-rattler-channel-string`; `ci/build_cpp.sh:28-33` notes that this keeps sccache working). The path is
    therefore fixed.
  * A change of the build path (or recipe name) would make the first restore after it useless, but it does no harm.

## 2. Why CAGRA is expensive

### What one link key is made of (standard dataset)

* **Search algorithm and its entry kernel:**
  * `search_single_cta` (+ `_p` persistent);
  * `search_multi_cta`;
  * multi-kernel: `random_pickup` + `compute_distance_to_child_nodes`, plus `apply_filter_kernel` when filtered;
  * multi-partition: `search_single_cta_mp`, `search_multi_cta_mp`.
* **Single-CTA sort variant.**
  * `topk_by_bitonic_sort` = `search_width·graph_degree ≤ 256`, and multi-warp merge only for itopk > 256
    (`search_single_cta.cuh:115,141`; `search_single_cta_kernel_launcher_common.cuh:38-50`).
  * In the tests that is always "bitonic, no multi-warp", except one itopk-512 case.
* **Metric.** `dist_op` is L2 (L2Expanded/Unexpanded), IP (also Cosine), L1, or Hamming (uint8 only). The
  normalization (cosine or no-op) makes Cosine a separate key from IP. That gives 4 metrics (5 for uint8).
* **(team size, dataset block dim):** (8,128), (16,256) or (32,512).
  * The instance is picked by the closest dim, or by an explicit `team_size` (`compute_distance_standard.hpp:51-58`,
    `compute_distance_standard_matrix.json`).
  * So dims ≤ ~191 → (8,128), ~192–383 → (16,256), larger → (32,512).
* **Filter:** none, bitset or bloom (used by `ann_cagra.cuh`); roaring (used by FILTER_UDF and
  CORE_ROARING_ALLOWLIST_TEST); or a UDF, keyed by its source text (FILTER_UDF only).
* **CAGRA-Q (VPQ):** (team, block dim, pq_len ∈ {2,4,8}, smem dtype).
* Data type and index/distance types are fixed per executable. `_U32` and `_I64` outputs share keys: the JIT kernels
  only use uint32 source indices.

### How many links (measured from the local cache, deduplicated by fragment names across library versions)

The fragment names in the NVVM keys reconstruct each link's fragment set. Across all executables in the suite, 650
distinct link keys are cached:

* IVF-PQ `compute_similarity`: 243
* CAGRA: 219
* IVF-Flat: 93
* pairwise distances: 71
* RaBitQ: 16
* IVF-SQ: 8

The cold suite run wrote 1,337 files, ≈ 650–800 links at 1–2 files each, so ≈ 1 s per cold link on average under
`ctest -j8` (696 s / ≈ 700).

CAGRA search links by data type (union over the executables that use that type):

| | single-CTA | single-CTA MP | multi-CTA | multi-CTA MP | multi-kernel (2 kernels) | total |
|---|---|---|---|---|---|---|
| float | 21 (14 standard: none/bitset/bloom; 4 VPQ; 1 itopk-512; 1 roaring and 1 UDF key from other executables) | 8 | 18 (1 roaring, 1 UDF) | 6 | 9 + 3 | 65 |
| uint8 | 18 | 8 | 19 | 6 | 7 + 3 | 61 |
| int8 | 14 | 8 | 16 | 6 | 7 + 3 | 54 |
| half | 5 (1 UDF) | 6 | 11 (1 UDF) | 6 | 4 + 3 | 35 |

* Float single-CTA example keys:
  * (8,128) bucket: L2/IP/Cosine/L1 with no filter; L2 and IP also with bitset and bloom (`ann_cagra.cuh:1149-1156`).
  * (16,256) and (32,512) buckets: L2/IP/L1.
  * VPQ: (16,256,pq 8) and (8,128,pq 8) × 3.
* Half has few keys because SetUp skips dims ≥ 256 (`ann_cagra.cuh:177,214-216`, finding 1 of
  `../PROFILING_SUMMARY.md`). Only the explicit-team-size cases reach the larger buckets.
* nsys per-process counts (earlier profiles, one process per gtest group) agree: ≈ 44–56 `cudaLibraryLoadData` per
  AnnCagraTest/IndexMerge shard process for float, 7 per multi-partition group, ≤ 5 per CAGRA-Q group.

### Cost per link

These are from the single-process cold and warm runs of `NEIGHBORS_ANN_CAGRA_FILTER_UDF_TEST` (`udf_s8/s9_{cold,warm}`
logs). Each case boundary is a log line.

| link | cold | warm |
|---|---|---|
| single-CTA search | 5.1–5.3 s | 20–40 ms per link (0.03–0.12 s per case, including the search) |
| multi-CTA search | 1.05–1.15 s | same |
| multi-kernel pair | 0.15–0.25 s | same |

* Single-CTA is 5× multi-CTA because the JIT single-CTA kernel dispatches at run time over every bitonic network
  size: `max_candidates` 64/128/256 × `max_itopk` 64/128/256/512
  (`jit_lto_kernels/search_single_cta_device_helpers.cuh:546-615`, called from `search_single_cta_jit.cuh:182-240`).
* So one link compiles all of them:

  | kernel | NVVM key | PTX | cubin |
  |---|---|---|---|
  | single-CTA | 198 KB | 824 KB | 603 KB |
  | multi-CTA | 171 KB | 303 KB | 245 KB |

### Model vs. measurement (keys × per-link cost, ignoring IVF-PQ graph-build links and cross-executable sharing)

| exe | model | measured cold − warm |
|---|---|---|
| float (own keys only: 19 single + 8 single-MP + 16 multi + 6 multi-MP + 12 MK) | ≈ 163 s | 145.6 s |
| uint8 (18 + 8 + 19 + 6 + 10) | ≈ 161 s | 143.9 s |
| int8 (14 + 8 + 16 + 6 + 10) | ≈ 137 s | 132.7 s |
| half (4 + 6 + 10 + 6 + 7) | ≈ 69 s | 67.3 s |

The model is somewhat high because some keys are linked first by another process in the same job, which then hits
(CAGRA_C_TEST, DYNAMIC_BATCHING, NEIGHBORS_TEST, TEST_BUGS, FILTER_UDF). In short: **≈ 60 links per executable, about
27 of them single-CTA links at ≈ 5 s, which make up ≈ 80% of the cost.**

## 3. Options

### a. CI: persist `CUDA_CACHE_PATH` across jobs (recommended; `change.patch`)

**Where the change goes.** In cuVS only.
* shared-workflows `conda-cpp-tests.yaml` already has `cache-paths`, `cache-key-prefix`, `cache-key-files`,
  `cache-key-matrix-fields`, `cache-environment` and `cache-read-only`. They were added in shared-workflows #628/#633
  (Sept 2026) and are implemented by `rapidsai/shared-actions/setup-caller-cache`.
* cudf uses them for its rtcx kernel cache (`NVIDIA/cudf` `.github/workflows/pr.yaml:376-401`, `test.yaml:58-84`,
  `ci/test_cpp_common.sh:36-43`).
* No shared-workflows change is needed, so `issue.md` is a cuVS issue.

**Design (and the pitfalls each choice avoids).**

1. `cache-environment: CUDA_CACHE_PATH=.cache/cuda-jit` and `cache-paths: .cache/cuda-jit`. `.cache` is already
   git-ignored, and the path is workspace-relative, as required for container jobs.
   * `ci/test_cpp.sh` makes the path absolute (`realpath -m`), because `ctest` runs after
     `pushd $CONDA_PREFIX/bin/gtests/libcuvs`. The driver would otherwise resolve a relative path per test process.
   * The script also logs the file count before and after the tests, so CI logs show the misses.
2. **The shard is part of the key prefix** (`libcuvs-cuda-jit-v1-shard<i>of<n>`).
   * The 4 shard jobs of one configuration share the reusable workflow's matrix, so without it they would compute the
     same key.
   * Only the first job to finish would save, so 3/4 of the entries would be lost. On the next run, the other shards
     would get an exact-key hit on a useless archive and never save.
3. **`cache-key-matrix-fields: CUDA_VER ARCH GPU DRIVER`.** The cache content depends on the GPU architecture, the
   nvJitLink version and possibly the driver; it does not depend on Python, the OS image or `DEPENDENCIES`.
   * Matching PR configurations against nightly ones (`prepare-matrix/matrix.yaml`):

     | key fields | PR configurations that can restore a nightly archive |
     |---|---|
     | default (7 fields) | 0 of 6 |
     | cudf's choice (6 fields, without PY_VER) | 1 of 6 |
     | without DEPENDENCIES as well | 3 of 6 |
     | without LINUX_VER as well (the 4 fields above) | **4 of 6** |

   * The two arm64 PR configurations (12.2.2/a100, 13.0.3/l4) have no nightly twin and stay cold.
   * Restoring a non-matching archive cannot give wrong results: entries are keyed by full content plus the nvJitLink
     stamp.
4. **`cache-key-files`** (`c/**`, `conda/recipes/libcuvs/**`, `cpp/**`, `dependencies.yaml`).
   * actions/cache keys are immutable. With no key files, the first archive would be restored by exact hit forever and
     never updated.
   * Hashing the C++ sources gives main a new key whenever libcuvs or its tests change. Each run first restores the
     most recent archive with the same prefix (`restore-keys`).
5. **PRs are read-only. `test.yaml` saves only when `inputs.branch == 'main'`.**
   * PR CI runs on pushes to `pull-request/N` branches, which can restore caches of their own branch and of the
     default branch, so they can use main's archives.
   * Archives saved by a PR would only be visible to that PR, and they would churn the repository cache quota.
6. **Size.** The default `CUDA_CACHE_MAXSIZE` (1 GiB) bounds each archive, and the driver evicts within it.
   * The data compresses very well with the `zstd --long=30` that actions/cache uses: the full local cache, 1.07 GB,
     becomes **26.5 MB** (40×); a 134 MB subset becomes 8.4 MB.
   * So a shard archive (≈ 100–300 MB uncompressed) is ≈ 5–25 MB. The 48 nightly archives (12 configurations × 4
     shards) are well under 1 GB.
   * Stale entries (fragments changed since) remain until LRU eviction. They cost only space. To reset, bump `v1`.

**Expected savings.**
* Per configuration with a warm archive: ≈ 696 s less test time (sum over the 4 shards). The single-job wall time
  drops from 1580 s to 856 s.
* The shards that hold the CAGRA executables get shorter by up to ≈ 2.5 min each: those executables take 80–170 s
  cold and 13–23 s warm.
* Per PR CI run: ≈ 4 × 11.6 ≈ **45 runner-minutes**, when the PR leaves the JIT fragments alone (docs, Python, tests,
  non-JIT C++).
* PRs that change a CAGRA header miss those links, as today (no regression).
* Nightly: ≈ 12 × 11.6 ≈ 2.3 runner-hours.
* The link cost is single-threaded CPU work, so these numbers scale with the CI runners' CPUs.

**Risks.** Low. A failed restore gives today's behaviour. Only trusted `main` runs write caches. A bad archive can be
dropped with `gh cache delete` or by bumping the prefix.

**Dependency.** The patch is against this branch, which already shards the C++ tests 4 ways (#2720) and uses
`matrix.shard`. Without #2720, drop `-shard…` from the prefix.

### b. Library

* **The driver cache already applies.** nvJitLink caches NVVM→cubin and PTX→cubin, so a warm cross-process link costs
  20–40 ms.
* **An rtcx on-disk cache of the final cubin** (`rtcx::cache_t`, keyed by fragment keys + fragment-content hash +
  nvJitLink version + arch):
  * It would only save the remaining 20–40 ms per warm link: ≈ 2 s per CAGRA executable, ≈ 20 s over the suite.
  * It would also make cuVS independent of `CUDA_CACHE_DISABLE` and of the shared 1 GiB driver-cache limit.
  * It does nothing for cold CI. Effort M, risk L–M (key completeness). Not recommended for CI time.
* **Cheaper single-CTA links (largest library-side lever, unmeasured).**
  * Make the bitonic sizes (`max_candidates`, `max_itopk`) compile-time parameters of the JIT search fragment, as in
    the non-JIT kernel, instead of runtime branches (`search_single_cta_device_helpers.cuh:546-615`).
  * Each link would then compile one network instead of ~10. The PTX would probably shrink from ≈ 824 KB to the
    multi-CTA scale (≈ 300 KB), so a cold link might take ≈ 1–2 s instead of 5 s.
  * There would be more keys when callers use several itopk/degree buckets. In the tests, `max_candidates` is 64
    everywhere (`search_width·graph_degree` ≤ 64) and `max_itopk` takes a few values (64, 128, 256, rarely 512).
  * Codegen changes (register allocation), so it needs search benchmarks.
  * Effort M–L, risk M. Users' first-search latency would benefit too.
* **Not promising:**
  * nvJitLink `-split-compile`: one entry kernel with everything inlined leaves little to split, and codegen could
    change.
  * Lower optimisation levels: these change kernel performance.
  * Runtime-dispatched metric/team size: this changes the hot loop.
* **Developers.** The local cache is at the 1 GiB default (1,073,408,166 B, 2,233 entries) and evicts.
  * One suite run at one libcuvs version needs ≈ 670 MB, so two versions thrash.
  * Suggest `CUDA_CACHE_MAXSIZE=4294967296` (the maximum) in the devcontainers (`.devcontainer/`).
  * This is not part of this patch.

### c. Tests

* The keys that CAGRA tests link are, by design, the product of metric × (team, block dim) × algorithm × filter. Each
  key is distinct device code: LTO inlines `dist_op`/`compute_distance` into each kernel.
* Ways to reduce them, and what they would cost in coverage:
  * test the 256/512 buckets with L2 only: ≈ 6 keys × ~6.4 s ≈ 38 s per data type;
  * run multi-partition single-CTA with L2 only: ≈ 4 × 5.1 s ≈ 20 s.
  * Both drop metric × team-size combinations, which is a coverage loss.
* With the CI cache in place these links are only paid when a PR changes the fragments, which is when the coverage
  matters most. **No test change recommended.**
* Already done on this branch: the single parameterised UDF (`26009a1d`, −3 UDF link sets).
* Keep each CAGRA executable in one process: the in-process cache needs it. Splitting executables across shards
  repeats shared links in each shard's archive, but not across runs.
* Side notes (coverage, not cost):
  * AUTO search resolves to MULTI_CTA in these tests, because `max_queries` = 10 < 2·num_SM
    (`search_plan.cuh:124-131`), so the AUTO blocks never exercise SINGLE_CTA.
  * Single-CTA is always "bitonic, no multi-warp" except one itopk-512 case, so the radix single-CTA variant is never
    linked by the CAGRA tests.

## 4. Recommendation

Apply `change.patch` (cuVS: `.github/workflows/pr.yaml`, `.github/workflows/test.yaml`, `ci/test_cpp.sh`).
`git apply --check` passes on this branch. yamllint (repo config), shellcheck (only the pre-existing SC1091 info) and
zizmor 1.24.1 `--offline` (same 0 findings / 23 ignored / 35 suppressed as before) pass on the patched copies. Nothing
was run on a GPU.

## 5. How to measure (parent, after the GPU is free)

All runs below are single-process; `KVIKIO_COMPAT_MODE=ON` as in CI. `Z=/home/coder/.conda/envs/rapids/bin/zstd`.

**M1. Cold, and restored from an actions/cache-style archive** (tar + `zstd --long=30` round trip), per executable.

```bash
cd /home/coder/cuvs/cpp/build/latest/gtests
export KVIKIO_COMPAT_MODE=ON Z=/home/coder/.conda/envs/rapids/bin/zstd
for exe in NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST NEIGHBORS_ANN_CAGRA_HALF_UINT32_TEST NEIGHBORS_ANN_IVF_PQ_TEST; do
  d=$(mktemp -d); r=$(mktemp -d)
  s=$(date +%s.%N); CUDA_CACHE_PATH=$d ./$exe --gtest_brief=1 > /dev/null; cold=$(echo "$(date +%s.%N) - $s" | bc)
  n=$(find $d -type f | wc -l)
  tar -C $d -cf - . | $Z -q -T0 --long=30 | $Z -q -d --long=30 | tar -C $r -xf -
  s=$(date +%s.%N); CUDA_CACHE_PATH=$r ./$exe --gtest_brief=1 > /dev/null; warm=$(echo "$(date +%s.%N) - $s" | bc)
  echo "$exe cold=$cold restored=$warm files=$n new_after_restore=$(( $(find $r -type f | wc -l) - n ))"
  python3 test-perf-proposal-reports/ci-jit-cache/count_links.py $d   # M4: links by kernel family
  rm -rf $d $r
done
```

* Expect "restored" ≈ the warm time (float ≈ 20 s), and `new_after_restore` ≈ 0.
* The C API test runs the same way via `ctest --test-dir /home/coder/cuvs/cpp/build/latest -R '^CAGRA_C_TEST$'` with
  `CUDA_CACHE_PATH` set.

**M2. Entries survive a rebuild at the same path.**

* Populate with one libcuvs build and replay with the next. Use snapshot s12 → s13: s13 only changes the KvikIO open
  path, so no JIT fragment changes.

  ```bash
  S=/tmp/claude-1000/-home-coder/ae252f72-9cc0-46ee-a543-8482b2c783fe/scratchpad; d=$(mktemp -d)
  CUDA_CACHE_PATH=$d LD_PRELOAD=$S/snap_s12/libcuvs.so $S/snap_s12/NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST --gtest_brief=1 >/dev/null
  n=$(find $d -type f | wc -l)
  /usr/bin/time -f "s13 with s12's cache: %e s" env CUDA_CACHE_PATH=$d LD_PRELOAD=$S/snap_s13/libcuvs.so \
    $S/snap_s13/NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST --gtest_brief=1 >/dev/null
  echo "new entries: $(( $(find $d -type f | wc -l) - n ))"   # expect ≈ 0
  ```

* None of the snapshots s6–s16 changes a JIT fragment, so any adjacent pair should give ≈ 0 new entries. A change to a
  fragment would add entries only for the link keys that contain it.

**M3. Suite.**

* `ctest -j8` (as in `timing_session4.sh`) with `CUDA_CACHE_PATH` set to a directory restored from a cold suite run's
  archive. Expect ≈ 856 s wall, against 1580 s cold.
* Also note the archive size: `tar -C $d -cf - . | zstd -T0 --long=30 | wc -c`. Expect tens of MB for the whole suite.

**M4. Exact link counts per executable** (Q2) from the M1 cold directories: `python3 count_links.py DIR` (in this
directory). It counts the NVVM-keyed entries by kernel family, and by data type for CAGRA. On a fresh cache that
count is the number of distinct links. It reads the fragment names from the `_INTERNAL_` symbols in each key.

**In CI after merging.**
* The first nightly on main should log `Cache saved with key: libcuvs-cuda-jit-v1-shard<i>of4-<CUDA>-<ARCH>-<GPU>-<DRIVER>-<hash>`.
* A later PR should log `Cache restored from key: …`.
* `ci/test_cpp.sh` should report nearly the same file count before and after the tests.

## 6. Uncertainties

* Per-executable link counts are lower bounds. They come from a full, LRU-evicting local cache spanning several
  library versions (deduplicated by fragment names), not from a fresh run. M4 gives the exact numbers.
* Per-link costs come from one executable run alone on this machine (i9-10980XE). Under `ctest -j8` the suite average
  was ≈ 1 s per link. CI runner CPUs differ.
* The nvJitLink two-level cache flow (NVVM-key hit skips everything; PTX-key hit skips ptxas) is inferred from the
  entry format and timings. It was not traced.
* Whether a driver upgrade invalidates the cache directory is unknown. `DRIVER` stays in the key, and runner driver
  updates can cause a one-off cold run.
* The GitHub cache scope rules assume that PR CI runs as pushes to `pull-request/N` branches (true for cuVS:
  `pr.yaml:3-5`), and that the nightly `test.yaml` runs on ref `main`.
