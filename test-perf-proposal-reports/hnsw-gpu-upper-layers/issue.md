### [PERF] HNSW GPU hierarchy runs NN-descent on tiny upper layers

**Problem**

The GPU hierarchy builds each upper HNSW layer's k-NN graph with `all_neighbors_graph`
(`cpp/src/neighbors/detail/hnsw.hpp`):

```cpp
// FIXME: choose better heuristic
bool use_nn_decent = neighbors.size() < 1e7;
```

Level L holds ~n / M^L points, and k = min(M, rows - 1). For most datasets that means a few dozen to a few thousand
rows, yet every layer goes through a full NN-descent build. Each call pays device and pinned allocations, an fp16
conversion and ~10+ iterations with host round trips. The result is still approximate, with fp16 distances.

This runs on every `hnsw::from_cagra` / `hnsw::build` with the default `HnswHierarchy::GPU`, on ACE serialization, and on
the layered `GRAPH_ONLY` artifact. In the HNSW-ACE test profiles, NN-descent on ~150-row levels took ~60 ms per call
(~2 s of profile per executable). `NEIGHBORS_HNSW_TEST` makes ~256 such calls.

**Proposal**

When a layer is small (n_rows² · dim below a fixed budget), compute its exact k-NN graph on the host: float distances,
OpenMP over rows, partial sort by (distance, id). Keep the NN-descent / IVF-PQ heuristic for larger layers.

* There are no GPU calls for these layers, and the results are exact and deterministic (ties go to the smaller id).
* Upper layers only choose the level-0 entry point, so search results change slightly and recall should not drop.
* No test hard-codes upper-layer graphs; all checks are recall thresholds.
