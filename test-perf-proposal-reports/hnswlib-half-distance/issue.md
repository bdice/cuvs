### [PERF] HNSW search on fp16 data rounds every element to fp16 in software and is 5-8x slower than int8

**Problem**

For `half` datasets, `hnsw::index_impl` uses `hnswlib::L2Space<half, float>` / `hnswlib::InnerProductSpace<half, float>`
(`cpp/src/neighbors/detail/hnsw.hpp:234-242`). These come from `cpp/cmake/patches/hnswlib.diff`. hnswlib's SIMD
kernels are float-only, so half takes the generic loops, which compute `half - half` (L2) or `half * half` (IP) with
the host `cuda_fp16` operators. On the host, each of those converts both operands to float in software, applies the
op, and rounds the result back to fp16 in software. Then `float(t)` converts it again. That gives:

* **Slow.** A distance call takes ~0.54 µs at dim 64 and ~8.6 µs at dim 1000 on one core, against ~0.012 / 0.23 µs
  for hnswlib's float SSE path. The library is built with `-march=nocona`, so nothing vectorizes. In
  `NEIGHBORS_ANN_HNSW_ACE_HALF_UINT32_TEST`, hnswlib search on half data is 5-8x slower than on int8/uint8: ≈ 4.6 s of
  profile, ≈ 3-4.5 s of ctest. The same path serves every user's half HNSW search, CPU-hierarchy build and `extend`.
* **Less accurate.** Every difference or product is rounded to fp16 before accumulation. The relative error of the
  distance is up to ~8e-4 at small dims (exact computation in float: ~1e-7).

**Proposal**

Give half its own hnswlib space in cuVS. Widen each element to float and accumulate in float:

* an exact, branch-free half→float conversion (vectorizable) for the portable kernel;
* an F16C kernel (`_mm256_cvtph_ps`) selected at run time with `__builtin_cpu_supports`, so the build flags stay
  unchanged and the binary remains portable;
* the same partial sums and reduction order in both kernels, so results do not depend on the CPU.

The data layout and serialized format are unchanged, and hnswlib.diff is untouched. CPU microbenchmark: 50-110x
faster per distance from dim 64 up, which is as fast as hnswlib's float path. Results change only by becoming more
accurate. The tests use recall thresholds, so no expected values change.
