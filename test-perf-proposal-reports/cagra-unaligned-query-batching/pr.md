### Batch CAGRA search queries regardless of row alignment

`search_main_core` searched queries in batches only when `dim * sizeof(T)` was a multiple of 16 B. For other dims,
and for queries passed with CAGRA row padding (the iterative graph build), it copied the queries and then ran one
plan call per query. This PR searches every input in batches of `max_queries`. The only code change is in
`cpp/src/neighbors/detail/cagra/cagra_search.cuh`, plus one comment in `cagra_build.cuh`. Kernels, JIT fragments
and the public API are unchanged.

* **Dense `[n, dim]` queries** are searched in place for any dim. This is the path aligned dims already took, and
  every dim took it before #1846. The kernels read query rows only in `setup_workspace`, with scalar loads at
  `queries + dim * query_id`, so the row pitch needs no 16 B alignment. Multi-partition search already relies on
  this. The full padded copy of the query set is gone.
* **CAGRA-padded `[n, stride]` queries** are packed one batch at a time into a dense `max_queries × dim` workspace
  buffer (`raft::copy_matrix`) and then searched as a batch. When the plan is persistent, the stream is synced after
  the copy, because the persistent runner doesn't wait on the caller's stream.
* **Same kernels and plan.** The per-query computation is the same. SINGLE_CTA (including persistent) is
  bit-identical to the old per-query path. MULTI_CTA's random entry points depend on the batch shape, so
  unaligned dims now get the same results that aligned dims always have.
* **Less memory for dense queries** (no copy). Padded queries take `min(max_queries, n) × dim` elements from the
  workspace resource.

## Testing

* All CAGRA test executables, `NEIGHBORS_TIERED_INDEX_TEST`, `NEIGHBORS_DYNAMIC_BATCHING_TEST`, `NEIGHBORS_MG_TEST`,
  the HNSW tests and `CAGRA_C_TEST` pass, as does the full `ctest` suite.
* `compute-sanitizer --tool memcheck` reports no errors on 145 CAGRA cases whose query rows are not 16-byte aligned
  (float, half and int8; every search algorithm and build algorithm, including iterative builds that pass
  CAGRA-padded queries) and on the iterative-build and NaN-query bug reproducers.

## Measurements

Single-process wall time of each test executable on an RTX 6000 Ada (48 GB) with a 36-core host, otherwise idle (no ctest parallelism, no MPS). Old and new builds were run alternately, 2 or more repetitions each with the order reversed between repetitions; mean (min–max).

The library changes on this branch were measured as a chain: each executable was run against six builds of `libcuvs.so` (before any of them, then after each one, cumulatively), alternating, 2 repetitions per build. For this PR, "before" is the build just before this change and "after" adds only this change; the test binaries are identical. Executables affected only by this change were measured separately in the same way.

| executable | tests before | tests after | before | after | change |
|---|---|---|---|---|---|
| `NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST` | 1417 | 1417 | 66.8 s (66.7–67.0) | 30.5 s (30.4–30.6) | -54.4% |
| `NEIGHBORS_ANN_CAGRA_HALF_UINT32_TEST` | 883 | 883 | 37.8 s (37.8–37.9) | 17.3 s (17.0–17.5) | -54.3% |
| `NEIGHBORS_ANN_CAGRA_INT8_UINT32_TEST` | 943 | 943 | 60.4 s (60.3–60.6) | 20.9 s (20.6–21.3) | -65.4% |
| `NEIGHBORS_ANN_CAGRA_UINT8_UINT32_TEST` | 943 | 943 | 63.0 s (62.6–63.4) | 21.7 s (21.4–21.9) | -65.6% |
| `NEIGHBORS_ANN_HNSW_ACE_FLOAT_UINT32_TEST` | 27 | 27 | 4.3 s (4.1–4.5) | 4.1 s (4.1–4.1) | -4.2% |
| `NEIGHBORS_ANN_HNSW_ACE_HALF_UINT32_TEST` | 25 | 25 | 6.3 s (5.9–6.8) | 6.2 s (5.8–6.6) | -2.0% |
| `NEIGHBORS_ANN_HNSW_ACE_INT8_UINT32_TEST` | 25 | 25 | 3.9 s (3.9–4.0) | 4.0 s (3.9–4.0) | +0.9% |
| `NEIGHBORS_ANN_HNSW_ACE_UINT8_UINT32_TEST` | 25 | 25 | 4.2 s (4.0–4.4) | 4.2 s (3.8–4.7) | +1.8% |
| `NEIGHBORS_ANN_CAGRA_BBQ_UINT32_TEST` | 135 | 135 | 1.9 s (1.8–1.9) | 1.9 s (1.9–2.0) | +4.3% |
| `NEIGHBORS_TIERED_INDEX_TEST` | 72 | 72 | 4.1 s (4.0–4.2) | 4.0 s (4.0–4.0) | -1.1% |
| `NEIGHBORS_ANN_CAGRA_TEST_BUGS` | 17 | 17 | 1.9 s (1.8–1.9) | 1.9 s (1.8–1.9) | -1.6% |
| `NEIGHBORS_ANN_CAGRA_MERGE_TEST` | 26 | 26 | 1.1 s (1.1–1.1) | 1.1 s (1.0–1.1) | +0.3% |
| `NEIGHBORS_ANN_CAGRA_FILTER_UDF_TEST` | 24 | 24 | 1.8 s (1.8–1.8) | 1.8 s (1.8–1.8) | +0.4% |
| `NEIGHBORS_DYNAMIC_BATCHING_TEST` | 281 | 281 | 30.7 s (30.5–30.9) | 30.6 s (30.3–30.8) | -0.4% |
| **total** | | | 288.2 s | 150.2 s | -47.9% |

All tests passed in every run.

Closes #`<issue>`
