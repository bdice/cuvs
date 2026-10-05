### Remove duplicate `npartitions` cases from HNSW ACE tests

ACE resolves `npartitions` 0 and 1 to 2 before doing anything else (`ace_resolve_partition_count` in
`cagra_build.cuh`). As a result, 24 of the 32 `AnnHnswAceBuild` cases in each HNSW ACE test executable ran the same
build three times.

This PR crosses only `npartitions = {2, 4}` with the other axes (dim × use_disk × metric). It adds one case for
`npartitions=0` (dim 64, in-memory, L2) and one for `npartitions=1` (dim 64, disk, IP), so the 0/1 → 2 resolution
path is still covered. Every distinct build is kept.

The change is test-only and touches only `cpp/tests/neighbors/ann_hnsw_ace.cuh`.

| | before | after |
|---|---|---|
| `AnnHnswAceBuild` cases per executable | 32 | 18 |
| tests, FLOAT | 41 | 27 |
| tests, HALF / INT8 / UINT8 (each) | 39 | 25 |

Parameter indices `/0`–`/17` now map to different inputs, and `/18`–`/31` no longer exist. No in-repo filters
reference them.

## Measurements

Single-process wall time of each test executable on an RTX 6000 Ada (48 GB) with a 36-core host, otherwise idle (no ctest parallelism, no MPS). Old and new binaries were run alternately, 2 repetitions each with the order reversed between repetitions; mean (min–max).

| executable | tests before | tests after | before | after | change |
|---|---|---|---|---|---|
| `NEIGHBORS_ANN_HNSW_ACE_FLOAT_UINT32_TEST` | 41 | 27 | 10.7 s (10.3–11.0) | 8.4 s (8.2–8.6) | -21.1% |
| `NEIGHBORS_ANN_HNSW_ACE_HALF_UINT32_TEST` | 39 | 25 | 15.5 s (15.1–16.0) | 11.2 s (10.8–11.5) | -28.0% |
| `NEIGHBORS_ANN_HNSW_ACE_INT8_UINT32_TEST` | 39 | 25 | 11.4 s (11.3–11.6) | 7.9 s (7.4–8.3) | -31.0% |
| `NEIGHBORS_ANN_HNSW_ACE_UINT8_UINT32_TEST` | 39 | 25 | 10.7 s (9.9–11.5) | 8.3 s (8.3–8.4) | -22.4% |
| **total** | | | 48.4 s | 35.8 s | -25.9% |

All tests passed in every run.

Closes #`<issue>`
