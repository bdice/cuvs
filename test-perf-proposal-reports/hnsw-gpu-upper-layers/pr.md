> **Status: rejected after measurement** (−1.2% on the HNSW executables, within noise). See `rejected.md`. This
> draft is kept for reference.

### Build small HNSW upper layers with an exact host k-NN

`all_neighbors_graph` (used by the HNSW GPU hierarchy, ACE serialization and the layered artifact) ran NN-descent on
every upper layer. Those layers are usually tiny (~n / M^L rows). NN-descent's per-call GPU setup dominated, and its
result is approximate.

When `n_rows² · dim ≤ 2²⁷`, this PR computes the layer's k-NN graph exactly on the host:

* widen the rows to float;
* compute squared L2 or the inner product against every other row (OpenMP over rows);
* partial-sort by (distance, id).

Self edges are never produced, every id is in range, and ties go to the smaller id. Larger layers keep the existing
NN-descent / IVF-PQ heuristic. A one-point level (k = 0) no longer calls NN-descent with `graph_degree = 0`.

Upper layers are now exact and deterministic. They only steer the search to the level-0 entry point, so search results
can change slightly, and recall should not drop. Tests check recall thresholds only, and no thresholds change.

## Testing

* `NEIGHBORS_HNSW_TEST` (256 cases), the four HNSW-ACE executables, `NEIGHBORS_ALL_NEIGHBORS_TEST`, BBQ and
  `HNSW_C_TEST` pass.
* A host harness checks the exact graph on 768 layer shapes (float, half, int8; L2 and inner product): every id is
  in range, there are no self edges, and the neighbor sets equal a brute-force reference.
* A first version that used GPU brute force crashed in `from_cagra` on tiny layers. This version computes the
  graph on the host and generates every id from the loop.

## Measurements

See `rejected.md`: 47.9 s → 47.4 s across `NEIGHBORS_HNSW_TEST` and the four HNSW-ACE executables (3 alternating
repetitions each).

Closes #`<issue>`
