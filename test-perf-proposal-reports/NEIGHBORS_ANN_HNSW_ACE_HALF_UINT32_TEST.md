# NEIGHBORS_ANN_HNSW_ACE_HALF_UINT32_TEST: where the time goes and how to make it faster

This uses the same fixture (`cpp/tests/neighbors/ann_hnsw_ace.cuh`), method and caveats as
[NEIGHBORS_ANN_HNSW_ACE_FLOAT_UINT32_TEST.md](NEIGHBORS_ANN_HNSW_ACE_FLOAT_UINT32_TEST.md), which explains the shared findings in full. Tests: `cpp/tests/neighbors/ann_hnsw_ace/test_half_uint32_t.cu` (39 cases; there are no FileIo or HnswAceWorkspace tests).
* The gtest sum under nsys is 27.4 s, or 24.4 s after removing 3 per-group copies of `CUFileInit`.
* ctest `-j8` took 15.8 s. That is the slowest of the four HNSW-ACE executables, **~4.7 s slower than int8/uint8**; see below. ctest/profile ≈ 0.65.

| group | cases | gtest s | GPU busy | dominant cost |
|---|---|---|---|---|
| AnnHnswAceTest_half.AnnHnswAceBuild | 32 | 19.2 | 6% | partition sub-builds 9.5 s; **half hnswlib search ~4.6 s** |
| AnnHnswAceMemoryFallbackTest | 1 | 3.4 | 4% | 30 partitions: 1.95 s; `CUFileInit` 0.99 s |
| AnnHnswInmemSpillTest | 2 | 2.5 | 1% | `CUFileInit` 0.91 s; 12 × buffer zero-fill 0.29 s; half searches |
| AnnHnswAceLayeredTest | 1 | 2.0 | 2% | `CUFileInit` 0.96 s |
| CagraAceWorkspaceHalf ×2, InvalidPartition | 3 | 0.3 | 0% | — |

## Where the time goes

* **Partition sub-builds: 9.5 s (39%).** NN-descent takes 5.3 s, the reverse-graph loop 2.3 s, and finish/prune 1.8 s. These are the same 80 builds as float (24 of the 32 cases are `npartitions` 0/1/2 triplicates). The float run was more heavily contended, which is why these numbers are lower here.
* **Half-specific: hnswlib CPU search/deserialize is 5-8x slower than for the other types.** Per 16 cases, compared with int8/uint8:
  * deserialize + search after the index copy: 2.07 s vs 0.23-0.26 s;
  * in-memory `from_cagra` + search: 2.40 s vs 0.45-0.65 s;
  * serialize + deserialize + search: 1.21 s vs 0.22-0.24 s.

  That is **+4.6 s (19%)**. In these windows the main thread runs ~90 ms per search with no CUDA calls or syscalls. hnswlib's SIMD path is only for `float` (`_deps/hnswlib-src/hnswlib/space_l2.h:229`, `space_ip.h:351`). `half` falls back to the generic `L2Sqr`/`InnerProduct` loop (`space_l2.h:6-21`), which evaluates `half - half` with host `cuda_fp16` operators: software half→float, float→half rounding, then half→float again for each element. The conda `-march=nocona` flags rule out F16C. The space is selected at `cpp/src/neighbors/detail/hnsw.hpp:234-241`. The ctest gap to int8/uint8 (4.7 s) matches.
* **Smaller shared costs:**
  * Upper-layer NN-descent: ~0.8 s.
  * `kvikio_ofstream` zero-fill: 0.80 s (28 opens × ~29 ms).
  * One real `CUFileInit` per process: 0.91-1.09 s.

## Ideas (ranked by estimated ctest savings out of 15.8 s; they overlap)

1. **Dedupe `npartitions` {0,1,2} (test-only, keeps coverage).** This is float idea 1. 14 cases, 7.0 s of profile → **~4.5 s (29%)**.
2. **Library: half distance kernels in hnswlib.** Do this in `cpp/cmake/patches/hnswlib.diff`. Accumulate in float with `__half2float` per element (no half arithmetic), or add an F16C/AVX path (`_mm256_cvtph_ps`) behind runtime dispatch. 4.6 s of profile → **~3-4.5 s (20-30%)**, or ~2 s after idea 1. Keeps coverage. Bonus: half L2/IP distances stop rounding each difference to fp16, and real users' half HNSW search gets faster. Effort low-medium; risk low (recall can only improve).
3. **Library: reverse-graph host path.** This is float idea 2 / CAGRA float idea 5. 2.3 s → **~1.5 s (9%)**.
4. **Skip `CUFileInit`.** This is float idea 3: **~1.0 s (6%)**. Reduces GDS coverage.
5. **Fallback test: fewer partitions.** This is float idea 6. Here it is 30 partitions, 1.95 s → **~0.9 s (6%)**.
6. **Stop the `kvikio_ofstream` zero-fill.** This is float idea 4: **~0.5 s (3%)**.
7. **Brute-force upper HNSW layers.** This is float idea 5: **~0.5 s (3%)**.

## Uncertainties

* The half-search diagnosis comes from cross-type phase comparison plus code reading; no CPU samples were available. The 5-8x figure is per phase, not a microbenchmark.
* The ctest scaling (0.65) is an average. The half search is single-threaded and probably less inflated by contention, so its ctest saving may be closer to the upper bound.
