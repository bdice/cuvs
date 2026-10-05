### Drop the redundant `reverse_batchsize` sweep from the Vamana tests

The degree-32 block of `NEIGHBORS_ANN_VAMANA_TEST` crossed `reverse_batchsize = {100, 1e6}` with all of its other
axes. `reverse_batchsize` only sets how `batched_insert_vamana` chunks the reverse-edge pass. The chunks hold
disjoint destination nodes, and each node's new edge list depends only on its own row, its reverse candidates and
the dataset. So the graph is the same for any value.

Existing sharded runs confirm this: all 480 `(100, 1e6)` pairs that started from the same `rand()` state logged
identical edge counts, including the 328 whose graph is not full. As a result, 112 of the 240 degree-32 cases per
data type rebuilt and re-checked a graph that was already checked.

This PR changes only `generate_inputs()` in `cpp/tests/neighbors/ann_vamana.cuh`:

* The degree-32 sweep (15 dims × visited_size × max_fraction × vamana_iters) uses `reverse_batchsize = 100`. With
  1000 rows, that runs several chunks per pass, including a short last chunk.
* Each of the 8 degree-32 configurations also runs once with `reverse_batchsize = 1e6` (one chunk), at dim 137.
* The degree 64/128/256 blocks are unchanged, and they run the one-chunk path at every dim.

Every data type × degree × visited_size × max_fraction × vamana_iters × dim combination is still built,
graph-checked, serialized and (for degree < 256, as before) recall-checked, with the same thresholds. The fixture
and the checks are unchanged.

The number of cases drops from 1,260 to 812 (315 → 203 per data type). `AnnVamana/<n>` indices after the
first removed case shift.

## Testing

* `NEIGHBORS_ANN_VAMANA_TEST` lists 812 cases instead of 1,260, and all pass.

## Measurements

Single-process wall time of each test executable on an RTX 6000 Ada (48 GB) with a 36-core host, otherwise idle (no ctest parallelism, no MPS). Old and new builds were run alternately, 2 or more repetitions each with the order reversed between repetitions; mean (min–max).

Both builds use the same `libcuvs.so`; only the test binary differs (1,260 → 812 cases).

| executable | tests before | tests after | before | after | change |
|---|---|---|---|---|---|
| `NEIGHBORS_ANN_VAMANA_TEST` | 1260 | 812 | 276.3 s (276.2–276.4) | 241.5 s (241.5–241.5) | -12.6% |

All tests passed in every run.

Closes #`<issue>`
