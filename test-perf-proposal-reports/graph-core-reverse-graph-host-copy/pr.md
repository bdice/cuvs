### CAGRA optimize: build the reverse graph from one device copy of a host graph

When the output graph of `cagra::graph::optimize` is host memory, `make_reverse_graph_gpu` used to do four steps for
every graph column: an OpenMP gather, a pageable H2D copy, a kernel and a stream sync. This is the default for
`cagra::build`, every ACE partition sub-build and BBQ builds. Each of these iterations costs 1.2–4.6 ms, while the
kernel itself takes microseconds.

This PR changes only `cpp/src/neighbors/detail/cagra/graph_core.cuh`:

* **Host graph:** copy the graph to the device once, with a single contiguous `raft::copy` from the large workspace
  resource, then run the same per-column launch loop as the device path. There is no host gather and no per-column
  sync.
* **Fallback:** if that allocation throws `std::bad_alloc`, the graph is staged in batches of columns sized to the
  free workspace memory (at least one column), with one gather, one copy and one sync per batch.
* **Memory:** the copy is the same size as `d_rev_graph` and comes from the same resource. `prune_graph_gpu` already
  stages the wider host kNN graph there, so with the default degrees (128 → 64) the peak of `optimize` is unchanged.
* **Results:** unchanged. Every path launches one `kern_make_rev_graph_k` per column, in column order, on one stream,
  with the same launch configuration. Column order decides which reverse edges are kept. `CagraOptimize.DeviceMatchesHost`
  covers host/device agreement.

## Testing

* `NEIGHBORS_ANN_CAGRA_HELPERS_TEST` (including `CagraOptimize.DeviceMatchesHost`), the four CAGRA, four HNSW-ACE,
  BBQ, merge and bug-reproducer test executables, and the full `ctest` suite pass.

## Measurements

Single-process wall time of each test executable on an RTX 6000 Ada (48 GB) with a 36-core host, otherwise idle (no ctest parallelism, no MPS). Old and new builds were run alternately, 2 or more repetitions each with the order reversed between repetitions; mean (min–max).

The library changes on this branch were measured as a chain: each executable was run against six builds of `libcuvs.so` (before any of them, then after each one, cumulatively), alternating, 2 repetitions per build. For this PR, "before" is the build just before this change and "after" adds only this change; the test binaries are identical. Executables affected only by this change were measured separately in the same way.

| executable | tests before | tests after | before | after | change |
|---|---|---|---|---|---|
| `NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST` | 1417 | 1417 | 84.2 s (83.4–84.9) | 77.5 s (77.2–77.9) | -7.9% |
| `NEIGHBORS_ANN_CAGRA_HALF_UINT32_TEST` | 883 | 883 | 47.9 s (46.7–49.0) | 44.9 s (44.6–45.2) | -6.2% |
| `NEIGHBORS_ANN_CAGRA_INT8_UINT32_TEST` | 943 | 943 | 71.6 s (71.3–71.9) | 67.8 s (67.2–68.5) | -5.3% |
| `NEIGHBORS_ANN_CAGRA_UINT8_UINT32_TEST` | 943 | 943 | 74.3 s (73.3–75.4) | 70.8 s (70.7–71.0) | -4.7% |
| `NEIGHBORS_ANN_HNSW_ACE_FLOAT_UINT32_TEST` | 27 | 27 | 7.6 s (7.3–7.9) | 6.7 s (6.3–7.1) | -11.1% |
| `NEIGHBORS_ANN_HNSW_ACE_HALF_UINT32_TEST` | 25 | 25 | 10.2 s (10.1–10.4) | 9.4 s (9.3–9.5) | -8.1% |
| `NEIGHBORS_ANN_HNSW_ACE_INT8_UINT32_TEST` | 25 | 25 | 7.4 s (7.4–7.5) | 7.3 s (6.7–7.9) | -1.6% |
| `NEIGHBORS_ANN_HNSW_ACE_UINT8_UINT32_TEST` | 25 | 25 | 7.6 s (7.5–7.7) | 7.1 s (7.0–7.1) | -6.7% |
| `NEIGHBORS_ANN_CAGRA_BBQ_UINT32_TEST` | 135 | 135 | 3.9 s (3.8–4.0) | 3.8 s (3.7–3.9) | -2.6% |
| **total** | | | 314.7 s | 295.4 s | -6.1% |

All tests passed in every run.

Closes #`<issue>`
