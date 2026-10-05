# NEIGHBORS_ANN_HNSW_ACE_UINT8_UINT32_TEST: where the time goes and how to make it faster

This uses the same fixture (`cpp/tests/neighbors/ann_hnsw_ace.cuh`), method and caveats as
[NEIGHBORS_ANN_HNSW_ACE_FLOAT_UINT32_TEST.md](NEIGHBORS_ANN_HNSW_ACE_FLOAT_UINT32_TEST.md), which explains the shared findings in full. Tests: `cpp/tests/neighbors/ann_hnsw_ace/test_uint8_t_uint32_t.cu` (39 cases).
* The gtest sum under nsys is 18.4 s, or 15.3 s after removing 3 per-group copies of `CUFileInit`.
* ctest `-j8` took 10.9 s; ctest/profile ≈ 0.72.
* This was the least-contended of the four profiles, so its proportions are the most trustworthy.

| group | cases | gtest s | GPU busy | dominant cost |
|---|---|---|---|---|
| AnnHnswAceTest_uint8_t.AnnHnswAceBuild | 32 | 11.4 | 10% | partition sub-builds 6.5 s |
| AnnHnswAceMemoryFallbackTest | 1 | 2.4 | 5% | `CUFileInit` 1.01 s; 28 partitions: 0.95 s |
| AnnHnswInmemSpillTest | 2 | 2.3 | 1% | `CUFileInit` 1.03 s; 12 × buffer zero-fill 0.34 s |
| AnnHnswAceLayeredTest | 1 | 2.0 | 2% | `CUFileInit` 0.99 s |
| CagraAceWorkspaceUint8 ×2, InvalidPartition | 3 | 0.35 | 0% | — |

## Where the time goes

* **Partition sub-builds: 6.5 s (42%).** NN-descent takes 3.9 s, the reverse-graph loop 1.05 s, and finish/prune 1.4 s. These are the same 80 builds of 5000 rows as float; 24 of the 32 cases are `npartitions` 0/1/2 triplicates. Even here the GPU is only 10% busy.
* **The reverse-graph loop shows how much contention distorts the profiles.** It does identical work in all four executables: 5120 single-column launches. Here it took 1.05 s (0.2 ms per iteration), against 6.4 s for float.
* **HNSW stage: ~2.3 s.**
  * Upper-layer NN-descent: ~0.75 s (0.37 s logged for the 16 disk cases).
  * `kvikio_ofstream` zero-fill: 0.82 s (28 opens × ~29 ms).
  * hnswlib search and I/O: ~15 ms per case.
* **Nothing uint8-specific stands out.** Its lower times compared with int8/float come from lighter host load during its profiling window, not from cheaper work.

## Ideas (ranked by estimated ctest savings out of 10.9 s; they overlap)

1. **Dedupe `npartitions` {0,1,2} (test-only, keeps coverage).** This is float idea 1. 14 cases, 4.1 s of profile → **~2.9 s (27%)**.
2. **Skip `CUFileInit`.** This is float idea 3: **~1.0 s (9%)**. Reduces GDS coverage.
3. **Library: reverse-graph host path.** This is float idea 2. 1.05 s → **~0.75 s (7%)**.
4. **Stop the `kvikio_ofstream` zero-fill.** This is float idea 4. 0.82 s → **~0.6 s (5%)**.
5. **Brute-force upper HNSW layers.** This is float idea 5. ~0.75 s → **~0.55 s (5%)**.
6. **Fallback test: ~9 instead of 28 partitions.** This is float idea 6: **~0.45 s (4%)**.

Ideas 1-6 together: ~5 s (~45%).

## Uncertainties

* The uncertainties are the same as for float. With the least host contention, these proportions are the best guide to what a lightly loaded CI machine would see.
