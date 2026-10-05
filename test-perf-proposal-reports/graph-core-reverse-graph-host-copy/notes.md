# graph-core-reverse-graph-host-copy

Library change, one file: `cpp/src/neighbors/detail/cagra/graph_core.cuh` (`make_reverse_graph_gpu`).
Patch: `change.patch` (applies to `test-perf-proposals` HEAD; `git apply --check` passes).
Hotspot: T7 in `../PROFILING_SUMMARY.md`, CAGRA float idea 5, HNSW ACE float idea 2, BBQ idea 6.

## What and why

`cagra::graph::optimize` builds the reverse graph in `make_reverse_graph_gpu`. When the output graph is
host memory (every regular `cagra::build`, every ACE partition sub-build, the BBQ build), the old code did this for
each of the `graph_degree` columns (`graph_core.cuh:836-852` before the patch):

1. an OpenMP `parallel for` gather of column k into a host vector,
2. a pageable H2D copy of that column,
3. one `kern_make_rev_graph_k` launch,
4. `sync_stream`.

The GPU work is µs-scale. The cost is the OpenMP fork/join, the pageable copy and the sync, 1.2–4.6 ms per column in
the profiles:

* CAGRA float: 14.6 s in 3144 iterations.
* HNSW ACE float: 6.4 s in 5120 iterations (80 sub-builds × 64 columns).
* BBQ: 0.84 s (32 columns per build).

The device path (`:828-834`) already launches one kernel per column back to back, with no copies or syncs.

## Change

* The per-column launch loop moves into a local lambda, `add_reverse_edges(view, n_cols)`. The device path and both
  host paths call it, so they all use the same kernel, launch configuration (1024×256), column order and stream.
* **Host graph, preferred path.** Allocate a `graph_size × degree` device copy from the large workspace resource. Copy
  the contiguous row-major host graph with a single `raft::copy`, then run the device-path loop on it. There is no
  OpenMP gather and no sync; `optimize()` already syncs after `make_reverse_graph_gpu`.
* **Host graph, fallback** (only if that allocation throws `std::bad_alloc`, which includes `rmm::out_of_memory`):
  * Stage batches of columns.
  * `batch_cols = min(max(workspace_free_bytes / (graph_size·sizeof(IdxT)), 1), degree)`. The buffer comes from the
    workspace resource, within its free budget. If not even one column fits there, a single column comes from the
    large workspace. The old code allocated a single column from the default device resource, so outside the
    workspace budget the fallback never needs more device memory than the old code did.
  * Host and device batch buffers are column-major. A partial batch is then one contiguous copy, and the kernel reads
    it coalesced.
  * Per batch: one OpenMP gather, one H2D, `n_cols` kernels and one sync. The sync is needed because the next gather
    reuses the host buffer. In the worst case (`batch_cols == 1`) this is the old algorithm.
* Includes added: `<raft/core/resource/device_memory_resource.hpp>`, which the file already used transitively, plus
  `<algorithm>`, `<new>` and `<optional>`.

## Memory

* The preferred path needs `graph_size × degree × sizeof(IdxT)` extra, the same size as `d_rev_graph` and from the
  same resource. Peak during the reverse step is `2 × N × degree × 4 B` (+ `N × 4 B` counts).
* `prune_graph_gpu` already stages the whole host kNN graph (`N × intermediate_degree × 4 B`) in the large workspace,
  and `intermediate_degree ≥ degree` is enforced.
* With the default 128 / 64 degrees, the peak of `optimize()` is therefore unchanged. It grows only when
  `intermediate_degree < 2 × degree`, by at most `(2·degree − intermediate_degree) × N × 4 B`.
* If that does not fit, the fallback takes over. Using `bad_alloc` as the signal is not exotic in this code base: ACE
  partition device reads (`cagra_build.cuh:1632`) and the Fastener merge (`cagra_merge.cuh:530`) do the same.
* Free-memory queries (`cudaMemGetInfo`) were avoided on purpose: with pool or async resources they under-report what
  the resource can actually serve.

## Why results are identical

* `kern_make_rev_graph_k` inserts all edges of column k (atomics on `rev_graph_count`) before column k + 1. When a node
  has more than `degree` incoming edges, this column order decides which reverse edges are kept.
* Every path keeps it: one launch per column, in increasing k, on the same stream. CUDA serialises same-stream
  launches, and these use no PDL attributes (the build also sets `-DCCCL_DISABLE_PDL`). The removed per-column host
  sync was not needed for ordering.
* Each launch reads the same values (`graph(i, k)`) with the same grid and the same grid-stride `src_id` mapping. Only
  the view type differs: full row-major device copy, or column-major batch, instead of a `[N, 1]` column.
* The only remaining nondeterminism is the hardware order of atomics *within* one column. It existed before, in both
  the host path and the device path, and is unchanged.
* After the patch, the host path runs exactly the device path's launch sequence. `CagraOptimize.DeviceMatchesHost`
  (`NEIGHBORS_ANN_CAGRA_HELPERS_TEST`) asserts element-wise equality of host and device `optimize` and should keep
  passing.

## Validation done (no GPU runs)

* Compiled `cpp/src/neighbors/cagra_optimize.cu` with its exact `compile_commands.json` command, changed as follows:
  * `-I…/cpp/src` and the source path pointed at a symlinked overlay that holds the edited `detail/cagra/` copy;
  * `-arch=sm_89` instead of the gencode list;
  * `nice -n 19`.

  `-E` output confirmed that only the overlay `graph_core.cuh` was included. The TU compiled with no warnings under
  `-Werror` (the only nvcc warning comes from `NVCC_PREPEND_FLAGS=-ccbin`). This TU instantiates both the host and
  the device `optimize`. Host symbols show both new kernel instantiations: the `layout_right` full copy and the
  `layout_left` batch.
* clang-format 20.1.8 with `cpp/.clang-format`: clean.
* `git -C /home/coder/cuvs apply --check change.patch`: OK.

Not done: running anything on a GPU. The fallback has no automated test. Ways to exercise it:

* Temporarily make the `try` body throw `std::bad_alloc{}`.
* Or give `optimize` a large-workspace resource wrapped in a limiting adaptor, with `intermediate_degree <
  2·degree` and a limit between `N·intermediate_degree·4` and `2·N·degree·4` bytes. Prune then fits but
  `d_rev_graph` plus the copy does not.

Then compare against `DeviceMatchesHost`.

## Tests to measure

Profiled host-graph `optimize` users (expected savings from T7, `n` = nsys s, `c` = ctest -j8 s):

| executable | expected saving |
|---|---|
| NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST | ~13 n (≈7 after test-index reuse) |
| NEIGHBORS_ANN_CAGRA_HALF_UINT32_TEST | ~10 n (≈5) |
| NEIGHBORS_ANN_CAGRA_INT8_UINT32_TEST | ~8 n (≈4) |
| NEIGHBORS_ANN_CAGRA_UINT8_UINT32_TEST | ~6 n (≈3) |
| NEIGHBORS_ANN_HNSW_ACE_FLOAT_UINT32_TEST | 1–2.4 c |
| NEIGHBORS_ANN_HNSW_ACE_HALF_UINT32_TEST | ~1.5 c |
| NEIGHBORS_ANN_HNSW_ACE_INT8_UINT32_TEST | ~0.9 c |
| NEIGHBORS_ANN_HNSW_ACE_UINT8_UINT32_TEST | ~0.75 c |
| NEIGHBORS_ANN_CAGRA_BBQ_UINT32_TEST | ~0.6 n |

Total ≈ 38 c, or ≈ 20 c after the CAGRA test-index reuse.

Correctness: `NEIGHBORS_ANN_CAGRA_HELPERS_TEST` (`CagraOptimize.*`, especially `DeviceMatchesHost`).

Also on this path but not profiled: `NEIGHBORS_ANN_CAGRA_TEST_BUGS`, `NEIGHBORS_ANN_CAGRA_FILTER_UDF_TEST`,
`NEIGHBORS_ANN_CAGRA_MERGE_TEST`, `NEIGHBORS_HNSW_TEST` and `NEIGHBORS_MG_TEST`, wherever they call `cagra::build`.

How to measure:

* Wall time per executable.
* Under nsys, the `cagra::graph::optimize(...)` NVTX range. Do not use `optimize/reverse`: after the patch the kernels
  are still running when that range closes, because the sync is in the caller.
* Expect the per-column `cudaStreamSynchronize` + `cudaMemcpyAsync` pairs inside `optimize/reverse` to disappear.

## Risks

* **Low.** Same kernel and same ordering, and the device path is already in production.
* Extra transient device memory in the reverse step, covered by the fallback (see Memory). If the large workspace is
  managed memory, the allocation succeeds and may oversubscribe, the same as `d_rev_graph`.
* Pageable H2D of the whole graph is one large synchronous-staging copy. That is the same bytes as before, in one
  transfer instead of `degree` transfers.
