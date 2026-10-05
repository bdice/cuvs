### Compute HNSW fp16 distances in float, with F16C when available

`hnsw::index` for `half` data used hnswlib's generic `L2Sqr` / `InnerProduct` templates (from `hnswlib.diff`). Those
evaluate `half - half` / `half * half` with the host `cuda_fp16` operators: software conversions plus a rounding to
fp16 for every element. That was slow (5-8x slower than int8 search) and needlessly inexact.

This PR adds `cpp/src/neighbors/detail/hnsw_half_distance.hpp` and a small `half_space` in `hnsw.hpp`:

* every element is widened to float and accumulated in float (16 partial sums, fixed reduction order);
* on x86-64 CPUs with AVX+F16C (checked once at run time) eight halves are converted per `vcvtph2ps`;
* other CPUs use an exact, branch-free conversion that auto-vectorizes. Both kernels return bit-identical results.

No build flags change, and hnswlib.diff and the index format are untouched. float/int8/uint8 are not affected.

CPU microbenchmark, one core, ns per distance (old → new):

| dim | L2 | IP |
|---|---|---|
| 64 | 542 → 10 | 696 → 10 |
| 128 | 1097 → 19 | 1354 → 16 |
| 1000 | 8571 → 112 | 10717 → 98 |

Max relative error against a double reference goes from ≤ 8e-4 to ≤ 3e-7. The fp16 → fp32 conversion was checked
against `__half2float` for all 65536 inputs.

Half search results can differ slightly because distances are now more accurate. Tests only check recall thresholds,
so no expected values change.

## Testing

* `NEIGHBORS_HNSW_TEST` (256 cases), the four HNSW-ACE executables, `HNSW_C_TEST` and the full `ctest` suite pass.
* The F16C and portable kernels agree bit for bit on 26,460 test pairs, and the half→float conversion matches
  `__half2float` for all 65,536 inputs.

## Measurements

Single-process wall time of each test executable on an RTX 6000 Ada (48 GB) with a 36-core host, otherwise idle (no ctest parallelism, no MPS). Old and new builds were run alternately, 3 or more repetitions each with the order reversed between repetitions; mean (min–max).

Both builds ran with `KVIKIO_COMPAT_MODE=ON`, 3 repetitions each. Only `libcuvs.so` differs.

| executable | tests before | tests after | before | after | change |
|---|---|---|---|---|---|
| `NEIGHBORS_HNSW_TEST` | 256 | 256 | 41.7 s (41.2–42.0) | 35.4 s (35.0–35.6) | -15.0% |
| `NEIGHBORS_ANN_HNSW_ACE_FLOAT_UINT32_TEST` | 27 | 27 | 3.3 s (3.2–3.6) | 3.2 s (3.1–3.4) | -2.5% |
| `NEIGHBORS_ANN_HNSW_ACE_HALF_UINT32_TEST` | 25 | 25 | 5.1 s (5.0–5.2) | 3.1 s (3.0–3.1) | -40.2% |
| `NEIGHBORS_ANN_HNSW_ACE_INT8_UINT32_TEST` | 25 | 25 | 3.0 s (3.0–3.0) | 3.0 s (3.0–3.1) | +1.3% |
| `NEIGHBORS_ANN_HNSW_ACE_UINT8_UINT32_TEST` | 25 | 25 | 3.0 s (3.0–3.1) | 3.0 s (2.9–3.1) | +0.1% |
| **total** | | | 56.1 s | 47.8 s | -14.9% |

All tests passed in every run.

Closes #`<issue>`
