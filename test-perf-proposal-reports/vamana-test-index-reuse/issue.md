### [TEST] Vamana tests build every degree-32 graph twice: `reverse_batchsize` doesn't change the graph

**Problem**

`NEIGHBORS_ANN_VAMANA_TEST` is the slowest C++ test executable: about 277 s in a single process on an RTX 6000 Ada.
That time is almost all Vamana index builds. It runs 4 data types × 315 cases from `generate_inputs()`
(`cpp/tests/neighbors/ann_vamana.cuh:310-388`), and every case builds and checks its own graph.

The degree-32 block (240 cases per data type) crosses `reverse_batchsize = {100, 1000000}` with 15 dims ×
`visited_size {64, 256}` × `max_fraction {0.06, 0.1}` × `vamana_iters {1.0, 1.5}`. `reverse_batchsize` only sets how the
reverse-edge pass of each insert batch is chunked (`batched_insert_vamana`, `cpp/src/neighbors/detail/vamana/vamana_build.cuh:499-555`):

* The pass splits the sorted, distinct destination nodes into chunks of `min(reverse_batchsize, N)`.
* For each destination, `populate_reverse_list_struct`, `recompute_reverse_dists`, `RobustPruneKernel`,
  `SortPairsKernel` and `write_graph_edges_kernel` read only that node's own graph row, its own reverse
  candidates and the dataset. They write only that node's row.
* The chunks share no nodes, so the graph after the pass does not depend on the chunking.
* The only randomness, `rand()` for the insert order and the medoid, doesn't depend on it either.

The existing logs confirm it. In a 4-shard run, case `i` goes to shard `i % 4`, so shards 0/2 and 1/3 hold the
rb 100 and rb 1e6 cases in the same order and from the same `rand()` state. All 480 such pairs logged the same
edge count, including all 328 whose graph is not full. By contrast, the same case started from a different `rand()`
state gives a different count in 649 of 652 non-full cases. Recall differed in 18 pairs, by at most 0.001, which is
the same noise the CAGRA search oracle shows between two runs of one binary.

So 112 of the 240 degree-32 cases per data type (448 builds in total, ≈ 37 s, 13% of the run) rebuild and
re-check a graph that their `reverse_batchsize = 100` sibling already checked.

No other redundancy exists: every case has distinct build parameters, and the search parameters are constant. So
an index cache like the ones in the CAGRA/IVF tests would never hit.

**Proposal (test-only)**

* Use `reverse_batchsize = 100` for the full degree-32 sweep. With `n_rows = 1000` that value takes the multi-chunk
  path, including a short last chunk.
* Keep `reverse_batchsize = 1e6` (one chunk) for each of the 8 degree-32 configurations at one dim (137).
* The degree 64/128/256 blocks, which all use 1e6, are unchanged. They cover the single-chunk path at every dim.

Every data type × degree × visited_size × max_fraction × vamana_iters × dim combination is still built,
graph-checked, serialized and (for degree < 256, as before) recall-checked with the same thresholds. Every (degree, visited_size,
max_fraction, reverse_batchsize, vamana_iters) build configuration still runs.

Cases: 1260 → 812. Expected time: ≈ 277 s → ≈ 240 s.
