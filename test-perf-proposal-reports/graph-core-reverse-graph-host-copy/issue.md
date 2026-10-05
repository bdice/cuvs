### [FEA] CAGRA `optimize`: reverse-graph step syncs once per graph column for host-resident graphs

**Problem**

`make_reverse_graph_gpu` (`cpp/src/neighbors/detail/cagra/graph_core.cuh:836-852`) handles an output graph in host
memory by doing four steps for each of the `graph_degree` columns:

1. an OpenMP `parallel for` gather of that column,
2. a pageable H2D copy of `graph_size` indices,
3. one `kern_make_rev_graph_k` launch,
4. `sync_stream`.

This is the path for every regular `cagra::build`, which optimizes into a host matrix. It is also the path for every
ACE partition sub-build and for BBQ builds.

The kernel takes microseconds. Each iteration is dominated by OpenMP fork/join, the pageable copy and the sync, at
1.2–4.6 ms. Profiles of the gtests (RTX 6000 Ada, nsys):

| executable | time in this loop | iterations |
|---|---|---|
| NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST | 14.6 s (~4% of the executable) | 3144 |
| NEIGHBORS_ANN_HNSW_ACE_FLOAT_UINT32_TEST | 6.4 s | 5120 (80 sub-builds × 64 columns) |
| NEIGHBORS_ANN_CAGRA_BBQ_UINT32_TEST | 0.84 s | 32 per build |

The HNSW ACE loop did identical work in four type profiles, yet took 1.05–6.4 s. That spread shows it is bound by
scheduling latency, not by compute.

The device-graph path right above it (`:828-834`) already launches the per-column kernels back to back, with no copy
and no sync.

**Proposal**

* Copy the host graph to the device once and run the existing device-path loop on the copy. The graph is contiguous
  and row-major, so this needs a single H2D and no host gather.
* The copy is the same size as `d_rev_graph` and comes from the same (large workspace) resource. `prune_graph_gpu`
  already stages the wider host kNN graph there, so with the default degrees the peak of `optimize` does not change.
* If the allocation fails, fall back to batches of columns sized to the free workspace memory. In the worst case
  that is one column per batch, as today.

Keep one launch per column, in order, on one stream. The order of the columns decides which reverse edges are kept
for nodes with more than `graph_degree` incoming edges, so results stay identical to the current host and device
paths.

Expected saving is ≈ 38 s of summed ctest time across the CAGRA, HNSW ACE and BBQ gtests. Large builds should also
benefit, because the strided host gather disappears and the `graph_degree` small copies become one large copy. That
was not measured.
