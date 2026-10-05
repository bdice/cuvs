# hnswlib-half-distance

Library change. The HNSW wrapper's host distance for `half` data stops rounding every element's difference or product
to fp16 in software. It now widens each element to float and accumulates in float, using F16C when the CPU has it.
`cpp/cmake/patches/hnswlib.diff` is **not** changed (see "Why not in hnswlib.diff").

Files: `cpp/src/neighbors/detail/hnsw.hpp` (modified) and `cpp/src/neighbors/detail/hnsw_half_distance.hpp` (new).
`git -C /home/coder/cuvs apply --check change.patch` passes, and the patch also applies on top of
`../hnsw-gpu-upper-layers/change.patch`.

## Where the time goes today

* `index_impl` picks the hnswlib space (`cpp/src/neighbors/detail/hnsw.hpp:234-242`). For `half` it uses
  `hnswlib::L2Space<half, float>` or `hnswlib::InnerProductSpace<half, float>`. Every hnswlib distance (search,
  `addPoint` for the CPU hierarchy, `extend`) goes through this space. Indexes loaded from disk use it too
  (`hnsw.hpp:325`).
* Those templates come from `cpp/cmake/patches/hnswlib.diff`. hnswlib's SIMD kernels are float-only (the patch wraps them in
  `if constexpr (std::is_same_v<DataType, float>)`, `_deps/hnswlib-src/hnswlib/space_l2.h:229`, `space_ip.h:351`). So
  `half` uses the generic loops:
  * L2 (`space_l2.h:6-21`): `DistanceType t = *pVect1 - *pVect2;`
  * IP (`space_ip.h:6-16`): `const DistanceType t = a[i] * b[i];`
  `half - half` and `half * half` are the host `cuda_fp16` operators (`cuda_fp16.hpp:202-203` → `__hsub`/`__hmul`,
  `:2623-2642`). Each one runs two software `__half2float`, a float op and a software `__float2half` with rounding
  (`:347, 526, 650, 685`: branchy bit manipulation; the subnormal path loops). Then `float(t)` runs one more
  `__half2float`. The loop cannot vectorize. Paths: conda CUDA 13.3 headers in `targets/x86_64-linux/include`.
* How it is compiled: `hnsw.hpp` is only included by `cpp/src/neighbors/hnsw.cpp`. The host compiler builds it
  (`x86_64-conda-linux-gnu-c++` 14.4, `-march=nocona -mtune=haswell -O3`), so there is no `-mavx`/`-mf16c`. hnswlib
  itself only gets `USE_SSE`.
* Effect in the tests (from `NEIGHBORS_ANN_HNSW_ACE_HALF_UINT32_TEST.md`): hnswlib search on half data is 5-8x slower
  than on int8/uint8. That is +4.6 s of profile (19%) and ≈ 3-4.5 s of ctest. The npartitions dedupe already on this
  branch cuts that to ~2 s. `NEIGHBORS_HNSW_TEST` (`AnnHNSW_H`, 32 cases, 500 queries, ef 250, dim up to 1000) was not
  profiled. Back-of-envelope: ~0.75 M distance calls per case at ~0.04-8.5 µs each gives ~100 CPU-seconds, or ≈ 3 s wall
  on 36 threads.

## The change

1. **`hnsw_half_distance.hpp` (new, no hnswlib dependency).** It holds hnswlib-style `DISTFUNC<float>` kernels for fp16:
   * `half_bits_to_float`: an exact, branch-free binary16 → binary32 conversion (normals, subnormals, ±0, inf, NaN). It
     uses a bitwise select so GCC vectorizes the element loop.
   * `half_distance_generic<InnerProduct>`: portable. It keeps 16 lane-wise float partial sums. The tail elements go to
     the first lanes. A fixed pairwise reduction tree finishes the sum. IP returns `1 - <a,b>` like hnswlib.
   * `half_distance_f16c<InnerProduct>`: `[[gnu::target("avx,f16c")]]`, `_mm256_cvtph_ps` (8 halves per
     instruction). It has the same 16 lanes, the same zero-padded tail and the same reduction tree, so it returns
     **bit-identical** results to the generic kernel (checked below). It uses no FMA.
   * `select_half_distance<InnerProduct>()` chooses F16C at run time
     (`__builtin_cpu_supports("avx") && __builtin_cpu_supports("f16c")`, cached). The F16C code is only compiled for
     x86-64 GCC/Clang host builds (`!__CUDACC__`). Everything else (aarch64, other compilers, nvcc front end) uses the
     generic kernel. The build flags are unchanged.
2. **`hnsw.hpp`.**
   * Includes the new header.
   * Adds `half_space<bool InnerProduct>`, a `hnswlib::SpaceInterface<float>` with
     `get_data_size() = dim * sizeof(half)`, the selected kernel, and `&dim_` as the parameter. Its data layout is
     identical to `L2Space<half,float>` and `InnerProductSpace<half,float>`.
   * `index_impl` uses `half_space<true/false>` for `T = half`. float/int8/uint8 are unchanged.

   The space class lives in `hnsw.hpp`, after the hnswlib includes, on purpose. If `<hnswlib/hnswlib.h>` is included
   before the CUDA headers, `-Werror=unused-function` fires on hnswlib's unused `static AVX512Capable()`. The existing
   include order relies on `cuComplex.h`'s leaked `#pragma GCC diagnostic ignored "-Wunused-function"`, and that
   behavior is left untouched.

### Why not in hnswlib.diff

hnswlib does not know the `half` type. Runtime CPU dispatch with target attributes is cuVS-specific glue. Changing the
patch also needs a fresh hnswlib fetch: an existing `_deps/hnswlib-src` keeps the old patch applied. The only other user
of hnswlib's generic half path is the HNSWLIB baseline in cuvs-bench (`cpp/bench/ann/src/hnswlib/hnswlib_wrapper.h:138,219`),
which this change does not touch. A follow-up could make that baseline use the same kernels.

## Measured on CPU (microbenchmark, not a test run)

Source: `half_distance_bench.cpp` (this directory; not part of the patch), built with the TU's flags
(`-march=nocona -O3`) and run on one core of the 36-thread host, `nice 19`. The host was shared, so the numbers are ±20%.
Pairs are consecutive random vectors with values in U(0.1, 2) (the HNSW test distribution).

| dim | L2 old (ns) | L2 new F16C | L2 new generic | IP old | IP new F16C | float hnswlib L2 (SSE), reference |
|---|---|---|---|---|---|---|
| 5 | 44 | 22 | 33 | 55 | 21 | 4.7 |
| 64 | 542 | 10 | 67 | 696 | 10 | 12 |
| 128 | 1097 | 19 | 124 | 1354 | 16 | 24 |
| 250 | 2149 | 48 | 248 | 2669 | 44 | 54 |
| 1000 | 8571 | 112 | 893 | 10717 | 98 | 230 |

* F16C is 2x faster at dim 5 and 50-110x faster from dim 64 up. Half distances are now as fast as hnswlib's float SSE
  path, or faster.
* The conversion is exact for all 65536 bit patterns (compared with `__half2float`; NaN compared as NaN).
* The generic and F16C kernels are bit-identical on 26,460 L2+IP pairs: dims 1-70, values N(0,1), N(0,1)·1e-5 (subnormals)
  and N(0,1)·100.
* Max relative error against a double-precision reference:

  | | dim 5 | dim 1000 |
  |---|---|---|
  | L2, old | 8.0e-4 | 5.9e-5 |
  | L2, new | 1.2e-7 | 2.9e-7 |
  | IP, old | 6.5e-4 | 2.6e-5 |
  | IP, new | 1.5e-7 | 1.8e-7 |

## Result differences

* Half L2 / IP distances returned by `hnsw::search` change in roughly the 4th significant digit, toward the exact value.
  IP products are now exact in float.
* Graph traversal compares these distances, so the visit order and the returned neighbors can change on near-ties.
  The same applies to graphs built on the host (`HnswHierarchy::CPU` build, `extend`).
* Recall should not drop (distances are strictly more accurate), but this is not a bitwise guarantee per query.
* The serialized format, `get_data_size()`, and float/int8/uint8 behavior do not change. Old half index files load and
  search with the new space.

## Affected tests and thresholds (nothing needs re-baselining)

* `NEIGHBORS_HNSW_TEST`, `AnnHNSW_H` (`cpp/tests/neighbors/hnsw.cu:120-177`): `eval_neighbours` with
  `min_recall = 0.98`, eps 0.006. `calc_recall` (`ann_utils.cuh:316+`) also counts a neighbor as matched when its
  distance is within eps of the reference. More accurate half distances can only help that match.
* `NEIGHBORS_ANN_HNSW_ACE_HALF_UINT32_TEST` (`ann_hnsw_ace.cuh`): `min_recall = 0.9` in every recall check
  (`:336, 377, 573, 620, 658, 872`).
* Python `test_hnsw.py` / `test_hnsw_ace.py` float16 cases: recall ≥ 0.9 / > 0.95 / ≥ 0.7.
* No test hard-codes half distances or neighbor IDs, so no expected values change.

## Risks

* `[[gnu::target]]` and `__builtin_cpu_supports` are GCC/Clang extensions. They are guarded, and MSVC/nvcc fall back to
  the generic kernel.
* aarch64 (Grace) uses the generic kernel: auto-vectorized, ~9x faster than before on x86. It was not compiled here
  because no aarch64 compiler is available. The code is plain C++17.
* If cuVS is built with `-march` that enables FMA, GCC may contract the generic kernel's `acc += x*y`. The results stay
  correct but may no longer be bit-identical to the F16C kernel. The default conda build (`nocona`) has no FMA.
* `cpu_supports_f16c()` uses a function-local static. It is thread-safe and runs once.

## Compile check

* `cpp/src/neighbors/hnsw.cpp` was compiled with its exact `compile_commands.json` command (`-Wall -Werror`, `-O3`,
  `-march=nocona`, `nice -n 19`). Only `-I.../cpp/src` was pointed at the overlay and `-o` at `$TMPDIR`. Exit 0, no
  warnings.
* `-M` confirms `/tmp/.../src/neighbors/detail/hnsw_half_distance.hpp` and the overlay `hnsw.hpp` were used. The object
  contains `half_distance_f16c<{true,false}>` with 16 `vcvtph2ps`, and `half_distance_generic<{true,false}>`.
* The new header also builds clean in isolation with `-Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror`.
* It compiles together with the hnsw-gpu-upper-layers patch.
* `clang-format` 20.1.8 (`cpp/.clang-format`) makes no changes.
* `hnsw.hpp` is included only by `cpp/src/neighbors/hnsw.cpp`, so that is the only affected TU.

## How to measure

Rebuild libcuvs (only `hnsw.cpp.o` changes; no test executable needs rebuilding). Then time:
* `NEIGHBORS_ANN_HNSW_ACE_HALF_UINT32_TEST` (main target; expect ≈ 2 s less after the dedupe already on this branch);
* `NEIGHBORS_HNSW_TEST` (half cases: `--gtest_filter='*AnnHNSW_H*'`; float/int8/uint8 cases are controls).

All tests should pass with the same thresholds.
