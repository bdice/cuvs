### NN-descent: run the per-iteration host update on the calling thread

`GNND::build` started and joined a `std::thread` on every iteration to run `update_graph` + `sample_graph` while the GPU ran
`add_reverse_edges` + `local_join`. Each of those threads is a new OpenMP root thread, so its parallel regions ran on a second OpenMP team
next to the caller's team, and the idle workers of one team spun (libomp's default `KMP_BLOCKTIME` is 200 ms) while the other team worked.
For small and medium graphs the iteration was dominated by thread start-up and oversubscription, not by GPU work: in the tests, 2.3–5.6 ms
per iteration against 0.3–1.1 ms of GPU time.

This PR changes only `cpp/src/neighbors/detail/nn_descent.cuh`. Both `build` overloads (dense and BBQ) now enqueue the iteration's GPU work
first, then call `update_and_sample` on the calling thread. The enqueue sequence is fully asynchronous, so the host work still overlaps the
GPU work. The two halves touch disjoint buffers, and the synchronization points are unchanged.

**Same output.** The host work and its inputs are unchanged; only the thread it runs on changes. Run-to-run variation from `local_join`'s
lock-ordered inserts is unchanged.

**Errors now propagate.** An exception thrown by `local_join` while the helper thread was joinable used to end in `std::terminate`. It now
propagates to the caller.

## Testing

* `NEIGHBORS_ANN_NN_DESCENT_TEST`, `NEIGHBORS_ALL_NEIGHBORS_TEST`, the CAGRA, HNSW-ACE, BBQ, filter-UDF, merge and
  bug-reproducer test executables, and the full `ctest` suite pass.

## Measurements

Single-process wall time of each test executable on an RTX 6000 Ada (48 GB) with a 36-core host, otherwise idle (no ctest parallelism, no MPS). Old and new builds were run alternately, 2 or more repetitions each with the order reversed between repetitions; mean (min–max).

The library changes on this branch were measured as a chain: each executable was run against six builds of `libcuvs.so` (before any of them, then after each one, cumulatively), alternating, 2 repetitions per build. For this PR, "before" is the build just before this change and "after" adds only this change; the test binaries are identical. Executables affected only by this change were measured separately in the same way.

| executable | tests before | tests after | before | after | change |
|---|---|---|---|---|---|
| `NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST` | 1417 | 1417 | 77.5 s (77.2–77.9) | 66.8 s (66.7–67.0) | -13.8% |
| `NEIGHBORS_ANN_CAGRA_HALF_UINT32_TEST` | 883 | 883 | 44.9 s (44.6–45.2) | 37.8 s (37.8–37.9) | -15.7% |
| `NEIGHBORS_ANN_CAGRA_INT8_UINT32_TEST` | 943 | 943 | 67.8 s (67.2–68.5) | 60.4 s (60.3–60.6) | -10.9% |
| `NEIGHBORS_ANN_CAGRA_UINT8_UINT32_TEST` | 943 | 943 | 70.8 s (70.7–71.0) | 63.0 s (62.6–63.4) | -11.1% |
| `NEIGHBORS_ANN_HNSW_ACE_FLOAT_UINT32_TEST` | 27 | 27 | 6.7 s (6.3–7.1) | 4.3 s (4.1–4.5) | -36.1% |
| `NEIGHBORS_ANN_HNSW_ACE_HALF_UINT32_TEST` | 25 | 25 | 9.4 s (9.3–9.5) | 6.3 s (5.9–6.8) | -32.6% |
| `NEIGHBORS_ANN_HNSW_ACE_INT8_UINT32_TEST` | 25 | 25 | 7.3 s (6.7–7.9) | 3.9 s (3.9–4.0) | -46.4% |
| `NEIGHBORS_ANN_HNSW_ACE_UINT8_UINT32_TEST` | 25 | 25 | 7.1 s (7.0–7.1) | 4.2 s (4.0–4.4) | -40.9% |
| `NEIGHBORS_ANN_CAGRA_BBQ_UINT32_TEST` | 135 | 135 | 3.8 s (3.7–3.9) | 1.9 s (1.8–1.9) | -51.2% |
| `NEIGHBORS_ANN_NN_DESCENT_TEST` | 916 | 916 | 52.6 s (51.9–53.7) | 33.7 s (33.4–33.8) | -36.0% |
| `NEIGHBORS_ALL_NEIGHBORS_TEST` | 228 | 228 | 28.3 s (27.4–29.2) | 20.8 s (20.6–21.0) | -26.4% |
| `NEIGHBORS_ANN_CAGRA_FILTER_UDF_TEST` | 24 | 24 | 3.2 s (3.0–3.5) | 1.8 s (1.8–1.9) | -41.8% |
| `NEIGHBORS_ANN_CAGRA_MERGE_TEST` | 26 | 26 | 1.6 s (1.6–1.8) | 1.1 s (1.1–1.1) | -33.4% |
| `NEIGHBORS_ANN_CAGRA_TEST_BUGS` | 17 | 17 | 2.7 s (2.6–2.8) | 1.9 s (1.9–2.0) | -28.2% |
| **total** | | | 383.9 s | 308.1 s | -19.7% |

All tests passed in every run.

Closes #`<issue>`
