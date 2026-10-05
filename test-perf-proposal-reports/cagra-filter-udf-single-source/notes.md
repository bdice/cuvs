# cagra-filter-udf-single-source

This is a test-only change to `cpp/tests/neighbors/ann_cagra/test_filter_udf.cu` (patch: `change.patch`, +27/−30). The
accept-all, reject-all, high-filtering and threshold UDF tests now share one UDF source and pick their predicate at
run time through `filter_data`. The tenant UDF stays a separate source.

## UDF mechanism

* API (`cpp/include/cuvs/neighbors/common.hpp:1611`): `filtering::udf_filter{source, void* filter_data, filtering_rate,
  function_name}`. The source defines `__device__ bool cuvs_filter_udf(uint32_t query_id, source_index_t source_id,
  void* filter_data)`. `filter_data` is an opaque, device-accessible pointer that is passed through unchanged as a
  kernel argument (`cagra_filter_payload.hpp:207`). It may be `nullptr`. **Runtime parameters are therefore
  supported**: the API has a user payload pointer.
* JIT (`src/neighbors/detail/cagra/jit_lto_kernels/sample_filter_udf.cuh:29-87`): the user source is wrapped into a
  `sample_filter<uint32_t>` specialization and compiled to LTO-IR with NVRTC. The fragment key is
  `"cagra_sample_filter_udf:uint32_t:<function_name>:<generated code>"`. It contains the full source text and no data
  type.
* Linking (`cagra_jit_launcher_factory.hpp`, rtcx `algorithm_planner.cpp:42-110`): each search algorithm has a
  planner. The planner adds static fragments (setup_workspace, compute_distance, search kernel, all tagged by data
  type) plus the UDF fragment. It then calls nvJitLink with `-lto -arch=sm_XX`. The in-process launcher cache key is
  the concatenation of all fragment keys, so each distinct (source, data type, algorithm) causes one nvJitLink LTO.
  SINGLE_CTA and MULTI_CTA each link one kernel and MULTI_KERNEL links two, so a (source, dtype) "link set" is 4 links.
  In the profile that was ≈ 2.0 s (float) and ≈ 2.2 s (half) with a cold cache, and ≈ 0.2 s with a warm cache.
* `filtering_rate` does not change which fragments are linked. Only MULTI_CTA uses it, and only to enlarge the runtime
  `itopk_size` (`search_plan.cuh:224-235`). The MULTI_CTA fragments have no itopk or `max_elements` template
  parameter (`search_multi_cta_planner.hpp`, `search_multi_cta_kernel_launcher_jit.cuh:63-81`). The SINGLE_CTA
  variants depend on itopk (64) and block size (256), which are the same in every test. So the per-test filtering
  rates do not create extra links after the merge. This resolves the MULTI_CTA caveat in the analysis.

## Distinct links, before → after (one process, i.e. one ctest run)

| | before | after |
|---|---|---|
| UDF (source, dtype) link sets | 6: float {accept_all, reject_all, high_filtering_rate, threshold, tenant}, half {threshold} | 3: float {min_source_id, tenant}, half {min_source_id} |
| UDF nvJitLink links (×4 per set) | 24 | 12 |
| successful NVRTC compiles (the key has no dtype) | 5 | 2 |
| failing NVRTC compiles (InvalidSourceThrows) | 3 | 3 |
| static-filter link sets (none / bitset / roaring, float) | 3 | 3 (unchanged) |

The expected saving is 3 link sets ≈ **6 s with a cold JIT cache**, out of the ≈ 15 s single-process run projected
in `NEIGHBORS_ANN_CAGRA_FILTER_UDF_TEST.md`. It is ≈ 0.6 s with a warm cache: 3 × ~0.2 s, mostly NVRTC plus cache
lookups. If CI's cache is fully cold, the 3 static-filter sets are also misses before and after the change, so the
absolute times rise but the difference stays about the same.

## Coverage argument

* All TEST_Ps and both instantiations are kept: 7 float tests and 1 half test, each × {SINGLE_CTA, MULTI_CTA,
  MULTI_KERNEL}, so 24 cases. Assertions and `filtering_rate` arguments are unchanged, and so are the effective
  search params.
* The filtered id sets are the same:
  * accept-all: `filter_data == nullptr` → `true`.
  * reject-all: min id = `n_rows`. The kernels only call the filter on valid ids `< n_rows`; invalid sentinels are
    skipped (`search_single_cta_jit.cuh:307-336`).
  * high filtering: min id = 704.
  * threshold: min id = 192, for both float and half.
  * tenant: unchanged.
* `ThresholdMatchesEquivalentBitset` still requires exact neighbor and distance equality with the bitset filter. It
  now also checks that the runtime parameter is read correctly by all three algorithms.
* What the tests exercise now, compared with before:
  * The UDF dereferences `filter_data` in 5 of 7 UDF searches; before, only tenant did.
  * `nullptr` `filter_data` is still passed (AcceptAll, RepeatedUdf).
  * Two different UDF sources are still linked for one dtype in one process (min_source_id and tenant), and one
    source for two dtypes (float and half). If a launcher or NVRTC cache ignored the source text or the dtype, the
    later test would run the wrong predicate and fail. For example, tenant run with the min_source_id kernel reads
    the context pointer as a threshold and fails `ASSERT_LT(source_id, n_rows)` or the tenant check. The tenant
    source carries a comment saying it is intentionally separate.
* Lost:
  * Predicates that are compile-time constants (`return true;` / `return false;`, `>= 192`), which LTO can fold.
    This is a codegen difference only; the API contract and results are the same.
  * Three extra distinct source texts. No test targeted these specifically.
* Not done, as a further option: folding tenant into the same source (a context struct with `min_source_id` plus
  optional tenant tables) would leave 2 link sets and save ≈ 2 s more. It would drop the two-sources-in-one-process
  check and make the tenant example less readable, so it is left out.

## JIT cache mechanism and how to measure with a cold cache

* rtcx keeps only in-process caches: the NVRTC LTO-IR map and the per-planner `launcher_jit_cache`. cuVS does not use
  rtcx's on-disk `rtcx::cache_t` (no references in `cpp/src` or `cpp/include`).
* The cross-process cache is the **CUDA driver JIT cache (ComputeCache)**, which nvJitLink uses internally:
  * `algorithm_planner::build()` does not pass `-no-cache`.
  * `libnvJitLink.so.13.4.92` `dlopen`s `libcuda.so.1` and calls `cuInit`/`cuGetExportTable`.
  * Its strings include "check cache for NVVM", "found entry in cache", "add ptx to cache", "check cache for PTX",
    "found cubin in cache", "add cubin to cache" and "no driver cache".
  * So both the LTO-IR→PTX and PTX→cubin steps are cached in the driver cache. Its location, size and on/off switch
    are controlled by `CUDA_CACHE_PATH` (default `~/.nv/ComputeCache`), `CUDA_CACHE_MAXSIZE` (default 1 GiB, and the
    local cache is full) and `CUDA_CACHE_DISABLE`.
  * This was inferred from the code and the binary, not traced. It agrees with the profile: cache-file mtimes matched
    the link ends.
* NVRTC (source → LTO-IR, ~75 ms per source) is never cached across processes.
* Measure, both builds, the whole executable in one process (sharding per test loses the in-process launcher
  cache):

  ```bash
  cd cpp/build/latest/gtests
  d=$(mktemp -d)
  CUDA_CACHE_PATH=$d ./NEIGHBORS_ANN_CAGRA_FILTER_UDF_TEST   # cold: gtest "(N ms total)"
  ls "$d" -R | wc -l                                          # entries written (fewer after the patch)
  CUDA_CACHE_PATH=$d ./NEIGHBORS_ANN_CAGRA_FILTER_UDF_TEST   # warm
  rm -rf "$d"
  ```

  Use a fresh `mktemp -d` for every cold run. `CUDA_CACHE_DISABLE=1` should also force misses, but that was not
  verified for nvJitLink.

## Verification done (no GPU)

* The TU compiles with its exact `compile_commands.json` command, with only these changes: `-arch=sm_89`, the source
  and `-I.../cpp/tests` pointed at an overlay copy, `-o` into `$TMPDIR`, and `-Werror` kept. The only output is nvcc's
  `compiler-bindir` notice, which comes from `NVCC_PREPEND_FLAGS` in the environment, not from the code.
* The new UDF source, wrapped exactly as `instantiate_cagra_sample_filter_udf()` wraps it, compiles with NVRTC using
  rtcx's options (`-arch=sm_89 -dlto -rdc=true --std=c++20 -default-device`) and produces 2,240 B of LTO-IR.
* `clang-format` 20.1.8 with `cpp/.clang-format` reports no changes. `git apply --check change.patch` passes.

## Risks

* Low, test-only.
* `raft::make_device_scalar(res, v)` copies H2D on `res`'s stream. The search uses the same stream and the scalar
  outlives the search, because it is declared before the `udf_filter` in the same scope.
* Verified on a GPU (commit 26009a1d): all 24 cases pass; cold-cache run 58.6 → 40.7 s, warm 1.8 → 1.4 s (see `pr.md`).
* With a warm cache (developer machines) the gain is small (~0.6 s). The gain matters for CI's cold cache.
