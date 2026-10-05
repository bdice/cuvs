# NEIGHBORS_ANN_HNSW_ACE_INT8_UINT32_TEST: where the time goes and how to make it faster

This uses the same fixture (`cpp/tests/neighbors/ann_hnsw_ace.cuh`), method and caveats as
[NEIGHBORS_ANN_HNSW_ACE_FLOAT_UINT32_TEST.md](NEIGHBORS_ANN_HNSW_ACE_FLOAT_UINT32_TEST.md), which explains the shared findings in full. Tests: `cpp/tests/neighbors/ann_hnsw_ace/test_int8_t_uint32_t.cu` (39 cases).
* The gtest sum under nsys is 21.4 s, or 18.2 s after removing 3 per-group copies of `CUFileInit`.
* ctest `-j8` took 11.0 s; ctest/profile ≈ 0.61.

| group | cases | gtest s | GPU busy | dominant cost |
|---|---|---|---|---|
| AnnHnswAceTest_int8_t.AnnHnswAceBuild | 32 | 14.0 | 8% | partition sub-builds 8.4 s |
| AnnHnswAceMemoryFallbackTest | 1 | 2.7 | 5% | 28 partitions: 1.22 s; `CUFileInit` 1.07 s |
| AnnHnswInmemSpillTest | 2 | 2.2 | 1% | `CUFileInit` 0.98 s; 12 × buffer zero-fill 0.30 s |
| AnnHnswAceLayeredTest | 1 | 2.1 | 2% | `CUFileInit` 0.97 s |
| CagraAceWorkspaceInt8 ×2, InvalidPartition | 3 | 0.35 | 0% | — |

## Where the time goes

* **Partition sub-builds: 8.4 s (46%).** NN-descent takes 5.3 s, the reverse-graph loop 1.5 s, and finish/prune 1.5 s. These are the same 80 builds of 5000 rows as float; 24 of the 32 cases are `npartitions` 0/1/2 triplicates.
* **The rest of the ACE build: ~1.7 s.** Labeling takes 0.78 s; the remainder is reorder, read and adjust.
* **HNSW stage: ~2.6 s.**
  * Upper-layer NN-descent: ~1.3 s (0.63 s logged for the 16 disk cases).
  * `kvikio_ofstream` zero-fill: 0.84 s (28 opens × ~30 ms).
  * hnswlib serialize/deserialize/search is fast, at ~15 ms per case: int8 uses `L2SpaceI` and the integer IP loop.
* **`npartitions=0` cases look slower** (4.6 s for 8 cases vs 2.4 s for `npartitions=1`). They are not: case /2 absorbs `CUFileInit` (1.23 s) and the first case absorbs warm-up.
* **Int8-specific:** only the Layered test's mixed-precision step (float-dataset deserialize + search, .cuh:583-629). It costs ~0.1 s and is not worth touching.

## Ideas (ranked by estimated ctest savings out of 11.0 s; they overlap)

1. **Dedupe `npartitions` {0,1,2} (test-only, keeps coverage).** This is float idea 1. 14 cases, 4.7 s of profile → **~2.9 s (26%)**.
2. **Skip `CUFileInit`.** This is float idea 3: **~1.0-1.2 s (9-11%)**. Reduces GDS coverage.
3. **Library: reverse-graph host path.** This is float idea 2. 1.5 s → **~0.9 s (8%)**.
4. **Brute-force upper HNSW layers.** This is float idea 5. 1.3 s → **~0.8 s (7%)**.
5. **Fallback test: ~9 instead of 28 partitions.** This is float idea 6: **~0.5 s (5%)**.
6. **Stop the `kvikio_ofstream` zero-fill.** This is float idea 4: **~0.5 s (5%)**.

Ideas 1-6 together: ~5 s (~45%).

## Uncertainties

* The uncertainties are the same as for float. The reverse-graph loop does work identical to float but took 1.5 s here vs 6.4 s there, which shows how much host contention distorts absolute numbers. Use the percentages.
