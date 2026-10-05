# hnsw-gpu-upper-layers

Library change in `cpp/src/neighbors/detail/hnsw.hpp`. When an upper HNSW layer is small (n_rows² · dim ≤ 2²⁷ ≈ 1.3e8
multiply-adds), its k-NN graph is now computed **exactly on the host** instead of with NN-descent on the GPU. That
covers every upper layer of the test datasets and the small top layers of real ones. Larger layers keep the existing
NN-descent / IVF-PQ path.

`change.patch` is against HEAD `594510e8`, which already contains hnswlib-half-distance.
`git -C /home/coder/cuvs apply --check change.patch` passes.

**This is the second version.** The first version used GPU `brute_force` and crashed. See "Crash in the first version"
below.

## Code and what it computes

`all_neighbors_graph<T>` (`hnsw.hpp:511-531` at HEAD) builds a k-NN graph of a small host matrix. For level L ≥ 1,
the input is the set of points whose HNSW level is ≥ L (≈ n / M^L rows), and k = min(M, rows - 1). The output (a uint32
host matrix, self excluded) becomes the level-L link lists. Today:

```cpp
// FIXME: choose better heuristic
bool use_nn_decent = neighbors.size() < 1e7;   // :517-518
if (use_nn_decent) { nn_descent::build(...) }   // graph_degree = k, intermediate = 2k
else { cagra::build_knn_graph(..., ivf_pq_params) }
```

Callers (HEAD line numbers):
* `from_cagra<GPU>` (`:1959`, call at `:2140`). This is the default `HnswHierarchy::GPU` path of `hnsw::from_cagra` and
  of in-memory `hnsw::build`. Every case of `NEIGHBORS_HNSW_TEST` goes through it. The ids are consumed unchecked by an
  OpenMP loop: `data[j] = order[neighbors[j] + start_idx]` (`:2157`).
* `serialize_to_hnswlib_batched` (`:1361`, call at `:1575`). This is the ACE disk / in-memory-spill serialization.
* `build_hnsw_upper_layer_graphs` (`:679`, call at `:706`). This is the layered `GRAPH_ONLY` artifact.

## Cost

* NN-descent pays a fixed setup cost on every call: device and 9 pinned allocations, fp16 conversion, and ~10+
  iterations, each with a blocking D2H copy and a host update. The inputs here are tiny.
* `NEIGHBORS_ANN_HNSW_ACE_FLOAT_UINT32_TEST.md` (idea 5): ~2 s of profile goes to NN-descent on ~150-row levels, 32 calls
  at ~60 ms each (≈ 0.7 s ctest). The summary (T9) puts it at ≈ 2.5 s ctest across the four HNSW-ACE executables.
* That profile predates `0bdf0a13` (the npartitions dedupe leaves 18 of 32 ACE builds) and `d3bbb9a8` (NN-descent's
  per-iteration host update runs on the calling thread). The remaining saving must be measured against HEAD. The
  estimate is ≤ 1.4 s ctest for the four ACE executables.
* `NEIGHBORS_HNSW_TEST` (not profiled) makes ~2 such calls per case: 128 cases, n = 1000/2000, M = 16/32, so level 1
  has ~31-125 rows and level 2 has ~1-8. That is about 256 calls.

## The change (version 2)

1. **New `exact_all_neighbors_graph<T>` (host only).** It widens the layer to float once (exact for half, int8 and
   uint8). Then, in an OpenMP loop over rows:
   * compute the squared L2 distance, or the negated inner product, to every other row in float;
   * map NaN to +inf so the ordering is total;
   * `std::partial_sort` the `(distance, id)` pairs, so ties go to the smaller id.

   Each row gets the first k ids. By construction, every id is in `[0, n_rows)`, differs from the row, and appears at
   most once. k = 0 (a one-point level, possible in `serialize_to_hnswlib_batched`) returns without work. `k < n_rows`
   and the row count are checked with `RAFT_EXPECTS` outside the parallel region.
2. **`all_neighbors_graph`:** if `n_rows² · dim ≤ 2²⁷`, use the host path. Otherwise the existing NN-descent / IVF-PQ
   heuristic (and its FIXME) is unchanged.

There are no GPU calls, allocations or syncs for these layers, and no new includes.

**Why this cap.** Tests:

| | rows | dim | multiply-adds |
|---|---|---|---|
| largest `NEIGHBORS_HNSW_TEST` layer | ~160 | 1000 | ~2.6e7 |
| ACE layers | ~156 | 128 | ~3.1e6 |

Real indexes: the top layers of a 1M-row, M = 32 index have ~1000 rows (1.3e8 at dim 128). At the cap the host does
~1.3e8 scalar multiply-adds: ~0.15 s on one core, a few ms on 36 threads. That is comparable to one NN-descent call.
Typical layers cost microseconds to ~1 ms.

## Crash in the first version (GPU brute force) and the fix

* **Symptom (coordinator's GPU run).** `NEIGHBORS_HNSW_TEST` and the ACE FLOAT/HALF tests died with SIGSEGV within
  ~1 s. The first crash was in AnnHNSWTest float with a tiny dataset (dim 5). gdb pointed to an OpenMP outlined region
  of `from_cagra<GPU>`. With only hnswlib-half-distance applied, all 256 `NEIGHBORS_HNSW_TEST` cases passed.
* **Where.** The only OpenMP loop that consumes the new output is the link-list fill at `hnsw.hpp:2157`,
  `data[j] = order[neighbors[j] + start_idx]`. It segfaults if any id is ≥ the layer size.
* **Version 1** wrote `static_cast<uint32_t>(id)` for whatever int64 id `brute_force::search` returned, with no range
  check. For L2 with k + 1 ≤ 64 that search takes the fused L2 k-NN kernel (`fused_l2_knn.cuh`). Unfilled slots of
  that kernel's warp-select heap keep the key `std::numeric_limits<uint32_t>::max()`, and a -1 from other paths becomes
  the same value after the cast. Either one becomes `order[4294967295 + start_idx]`.
* **Root cause not confirmed.** I could not run a GPU here, and on paper `brute_force` should fill every slot when
  k + 1 ≤ n. The search used the same matrix as queries and index, and the layers were tiny (top levels have k + 1 == n,
  e.g. 4 rows with k = 3, dim 5). Those cases are not covered by the brute-force tests.
* The post-processing logic itself was re-checked on the host (see below). With valid search output it cannot produce
  an out-of-range id, so the bad ids must have come from the GPU search output.
* **Fix.** Version 2 does not depend on GPU search output for these layers. It computes them on the host, where every id
  is generated from the loop index. This also removes the per-layer device allocation, H2D/D2H copies and syncs, and
  the possible first use of cuBLAS (inner product) that version 1 added.

## Host verification (no GPU)

* **Python mirror of the patched code** (`/tmp/hnswfix/upper2/sim.py`). It reproduces `from_cagra<GPU>`'s level loop:
  hnswlib-style random levels, bucket sort, `neighbor_size`, the `num_pts <= 1` skip, and the
  `order[neighbors[j] + start_idx]` mapping. It ran 116 layers: n_rows 1000/2000, dim 5/64/250, M 16/32, L2/IP,
  float/fp16/int8. Every id was in range, with no self edges and no duplicates, and the neighbor sets equal a numpy
  exact reference.
* **C++ harness** (`/tmp/hnswfix/upper2/harness.cpp`). It includes the patched `hnsw.hpp` and checks
  `exact_all_neighbors_graph` for float/half/int8/uint8, n 2-160 (including k = n - 1), dim 1-250, L2/IP and k = 0. It
  checks ranges, self edges, duplicates, and that the j-th returned distance equals the j-th smallest. It is **not yet
  compiled or run**, because a timing session was running (see below).

## Result differences

* Upper layers become the exact k-NN graphs (float distances). NN-descent was approximate and computes distances in
  fp16 for dim > 16 and for all non-float inputs (`nn_descent.cuh:2412-2423`).
* Ties are broken by the smaller id, so the result is deterministic. Layers with ≤ M + 1 rows were already complete
  graphs; their neighbor sets are unchanged, but the order may differ.
* Upper layers only steer the greedy descent to the level-0 entry point, so final search results can change slightly.
  Recall is expected to be equal or better. The base layer (CAGRA graph) is unchanged.

## Affected tests and thresholds (nothing needs re-baselining)

* `NEIGHBORS_HNSW_TEST`: all cases use the GPU hierarchy, with `min_recall = 0.98` and eps 0.006.
* `NEIGHBORS_ANN_HNSW_ACE_{FLOAT,HALF,INT8,UINT8}_UINT32_TEST`: `min_recall = 0.9`.
* `HNSW_C_TEST` and Python `test_hnsw*.py`: recall thresholds only.
* No test compares upper-layer link lists with stored values.

## Risks

* **Host CPU time at the cap.** Up to ~1.3e8 multiply-adds per layer, parallelized with OpenMP. On a single-threaded
  host near the cap it is ~0.15 s per layer, about one NN-descent call.
* **Default OpenMP team.** The loop uses the default team size, like other unparameterized loops in this file. It does
  not use `index_params::num_threads`, because `all_neighbors_graph` does not receive it.
* **GPU result.** This version passes all HNSW tests, but measured only −1.2% on the HNSW executables, so the
  proposal was rejected (see `rejected.md`).

## Compile check

Version 2 was compiled later: `hnsw.cpp` builds cleanly with `-Wall -Werror`, the host harness passed 768 layer shapes with no bad entries, and the library build and the HNSW tests passed on the GPU.
* `clang-format` 20.1.8 makes no changes.
* The patch applies to HEAD (`git apply --check`) and round-trips to the overlay.
* `/tmp/hnswfix/upper2/run_checks.sh` does the rest. It refuses to run while the benchmark is active. When free, it:
  * compiles `cpp/src/neighbors/hnsw.cpp` with its exact `compile_commands.json` flags (`-Wall -Werror`, overlay `-I`,
    `-o` in `/tmp`, `nice -n 19`);
  * prints the `-M` header list;
  * builds and runs the host harness.

  The version-1 compile check passed but does not apply to this version.

## How to measure

Rebuild libcuvs (only `hnsw.cpp.o` changes). Run:
* `NEIGHBORS_HNSW_TEST`;
* `NEIGHBORS_ANN_HNSW_ACE_{FLOAT,HALF,INT8,UINT8}_UINT32_TEST`.

All tests should pass with unchanged thresholds. The `nn_descent` NVTX ranges inside `hnsw::from_cagra<GPU>` and the
ACE serialization should disappear for these layers.
