### Use grid-stride loops in the IVF-PQ codepacker kernels

`write_list`, `write_list_flat` and `run_on_list` (`ivf_pq_codepacking.cuh`) started at the global row index but
stepped by one block's worth of rows. Their launchers already cover all rows with the grid, so block `b` re-processed
every row from `b * rows_per_block` to the end of the list. The work was therefore quadratic in the list size, and
block 0 encoded `ceil(n_rows / 8)` rows serially.

This hit `encode_list_data` (`codepacker::extend_list`) hardest. At `pq_dim = 3072` a 128-row list took 127–146 ms per
launch, and `NEIGHBORS_ANN_IVF_PQ_TEST` spent 26 s in this kernel. `pack/unpack_list_data`,
`{un}pack_contiguous_list_data` and `reconstruct_list_data` have the same problem for lists longer than 256 rows.

The fix makes the three loops grid-stride loops: the stride is multiplied by `gridDim.x`.

* Launch configurations are unchanged.
* The per-row computation (`encode_vectors`, the same lane layout and reduction order) is unchanged, so the written codes
  are bit-identical. Each row is just processed once instead of up to `n_rows / rows_per_block` times.
* This also removes a possible lost-update race. With `pq_bits < 8`, the FLAT-layout encoder and `unpack_contiguous`
  do bitfield read-modify-writes, and duplicated rows could collide.

Expected: `encode_list_data` gets ≈ `ceil(n_rows / 8)`× faster per launch (≈ 16× in the tests, ≈ 8 ms at `pq_dim` 3072),
and pack/unpack on large lists becomes linear. The main `build`/`extend` path is unaffected.

## Testing

* All IVF-PQ executables and the full `ctest` suite pass. Codes are computed by the unchanged per-row routine;
  only the number of times each row is processed changes (once instead of up to 16 times).

## Measurements

Single-process wall time of each test executable on an RTX 6000 Ada (48 GB) with a 36-core host, otherwise idle (no ctest parallelism, no MPS). Old and new builds were run alternately, 2 or more repetitions each with the order reversed between repetitions; mean (min–max).

The IVF changes on this branch were measured as a chain: each executable was run against the builds before and after each of them (cumulatively), alternating, 2 repetitions per build. For this PR, "before" is the build just before this change and "after" adds only this change. The other IVF-PQ users were measured separately in the same way. Only `libcuvs.so` differs.

| executable | tests before | tests after | before | after | change |
|---|---|---|---|---|---|
| `NEIGHBORS_ANN_IVF_FLAT_TEST` | 390 | 390 | 82.8 s (71.7–94.0) | 82.9 s (70.5–95.3) | +0.1% |
| `NEIGHBORS_ANN_IVF_PQ_TEST` | 1065 | 1065 | 158.6 s (157.9–159.4) | 138.5 s (138.2–138.7) | -12.7% |
| `NEIGHBORS_ANN_IVF_SQ_TEST` | 134 | 134 | 14.9 s (14.9–14.9) | 14.8 s (14.8–14.8) | -0.5% |
| `NEIGHBORS_MG_TEST` | 15 | 15 | 13.8 s (13.7–13.8) | 13.8 s (13.8–13.8) | +0.0% |
| `NEIGHBORS_TIERED_INDEX_TEST` | 72 | 72 | 3.0 s (3.0–3.0) | 3.0 s (3.0–3.0) | -0.8% |
| `NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST` | 1417 | 1417 | 24.0 s (24.0–24.0) | 24.2 s (24.1–24.3) | +0.7% |
| `NEIGHBORS_ANN_CAGRA_HALF_UINT32_TEST` | 883 | 883 | 14.2 s (14.2–14.2) | 14.3 s (14.2–14.3) | +0.6% |
| `NEIGHBORS_ANN_CAGRA_INT8_UINT32_TEST` | 943 | 943 | 16.7 s (16.6–16.9) | 16.8 s (16.7–16.8) | +0.1% |
| `NEIGHBORS_ANN_CAGRA_UINT8_UINT32_TEST` | 943 | 943 | 17.6 s (17.6–17.7) | 17.5 s (17.4–17.5) | -0.9% |
| `NEIGHBORS_DYNAMIC_BATCHING_TEST` | 281 | 281 | 30.4 s (30.4–30.4) | 30.8 s (30.8–30.9) | +1.5% |
| `NEIGHBORS_ALL_NEIGHBORS_TEST` | 228 | 228 | 16.8 s (16.7–16.9) | 16.8 s (16.8–16.9) | -0.0% |
| **total** | | | 392.9 s | 373.3 s | -5.0% |

All tests passed in every run.

Closes #`<issue>`
