# cagra-unaligned-query-batching

Library fix for hotspot T3 in `../PROFILING_SUMMARY.md`: CAGRA search runs one plan call per query when the
query row pitch is not a multiple of 16 B (float `dim % 4`, half `% 8`, int8/uint8 `% 16`) or when queries are
passed with CAGRA row padding. Patch: `change.patch` (applies to `test-perf-proposals`; touches
`cpp/src/neighbors/detail/cagra/cagra_search.cuh` and one comment in `cagra_build.cuh`).

## Why the per-query loop exists

`search_main_core` (`cagra_search.cuh:94-165` before the patch) does this:

| queries passed | what happened | batched? |
|---|---|---|
| dense `[n, dim]`, `dim * sizeof(T) % 16 == 0` | view | yes |
| dense `[n, dim]`, pitch not 16 B aligned | `make_device_padded_dataset` copy (memset + 2D copy of the whole query set) to `[n, stride]` | **no**, one plan call per query |
| CAGRA-padded `[n, stride]` (`stride > dim`, accepted by `search_main`; used by the iterative build) | view | **no**, one plan call per query |

Every kernel reads its query row through the JIT fragment `setup_workspace(desc, smem, queries_ptr, query_id)`,
which advances with `queries_ptr += dim * query_id` (`jit_lto_kernels/setup_workspace_impl.cuh:53,143`). `dim` is
the logical dim from the dataset descriptor. A batch therefore needs a row pitch of exactly `dim`. The padded copy
has pitch `stride != dim`, so the code fell back to `n_queries = 1` launches with the row selected by the base
pointer.

The 16 B rule was introduced by #1846 ("New Dataset API Clarifying Ownership", commit `2db820e7`, merged through
#2392). Its comment says dense rows "can be misaligned between rows (e.g. float, dim=1) and trigger misaligned
access in CAGRA search". The device code doesn't support that:

* The only reads of the global query buffer are in `setup_workspace_standard_impl` (`buf[j] = mapping(queries_ptr[i])`)
  and `setup_workspace_vpq_impl` (`mapping(queries_ptr[i])`, `queries_ptr[i + 1]`, `queries_ptr[i + k]`). These are
  scalar `DATA_T` loads. They need only `sizeof(T)` alignment, and the compiler can't merge them into vector loads
  because `dim` is a runtime value. The 16 B vectorized (`LOAD_T`) loads are on the **dataset** rows, which is why the
  dataset must be padded.
* Every search kernel gets the query through this function: single-CTA (normal, persistent, multi-partition),
  multi-CTA (normal, multi-partition), and multi-kernel (`random_pickup`, `compute_distance_to_child_nodes`). No
  other device code reads the query buffer. Cosine normalisation reads only dataset norms.
* Before #1846, `search_main_core` passed `queries.data_handle() + dim * qid` and batched every dim. The tests at that
  time already covered dims 1, 3, 5, 7, 17, 137 and 619.
* `search_multi_partition` (same file) still passes dense `[n, dim]` queries to the same kernels in batches. The
  int8/uint8 `AnnCagraMultiPartitionTest` covers this with dim 8, which has an 8 B pitch.

## Chosen approach: dense batches, no kernel changes

The rewritten `search_main_core` loop does the following:

* **Dense `[n, dim]` queries (any dim)** are searched in place, one plan call per `max_queries` batch. This is the
  same code path that aligned dims already take, and it no longer copies anything. The old code copied the whole
  query set.
* **CAGRA-padded `[n, stride]` queries** are packed one batch at a time into a dense `max_queries x dim` buffer with
  `raft::copy_matrix` (`cudaMemcpy2DAsync`, D2D). The batch is then searched with one plan call. The buffer is a
  `lightweight_uvector` taken from the workspace resource (the same allocator the plans use) and reused for every
  batch.
* **Persistent mode with padded input:** the persistent runner takes jobs from the host without waiting on the
  caller's stream, so the stream is synced after the pack copy. This applies only when `plan->persistent` is set.
  The old padded-copy path had the same race and did not sync.
* The device pointer is still resolved with `expect_device_accessible_data_handle`. Aligned and padded inputs used to
  go through the same call inside `make_device_padded_dataset_view`.
* The output pointer offsets are now computed in `size_t`. The old batched path multiplied `topk * qid` in 32 bits.

Kernels, JIT fragments, `.cu.in` templates, the matrix JSONs, launchers and plan signatures are unchanged.

### Alternatives considered

1. **Pass a query leading dimension into `setup_workspace`** (the T3 suggestion). This removes the pack copy for
   padded input, but it changes the extern fragment signature: `extern_device_functions.cuh`,
   `setup_workspace_kernel.cu.in` and both `setup_workspace_*_impl`. It also changes every kernel that calls the
   fragment, with their `.cu.in` wrappers: single-CTA, single-CTA persistent (its job descriptor needs the ld),
   single-CTA mp, multi-CTA, multi-CTA mp, `random_pickup` and `compute_distance_to_child_nodes`. On top of that it
   changes the host kernel function typedefs and launchers and `search_plan_impl::operator()` in all three plans,
   roughly 20 files. The only extra saving is one D2D copy of `batch x dim` elements. Only the iterative build passes
   padded queries, and it already copies or reconstructs each 8192-row chunk, so the extra copy is negligible
   (for example 4.5 MB for float dim 137).
2. **Copy into a 16 B-aligned padded buffer and batch** (the "simpler alternative" in the task). This doesn't work on
   its own. The old code already makes exactly this copy, and the copy is what forces the per-query loop, because the
   kernels stride by `dim`, not by the padded pitch. Without kernel changes, a batch has to be dense.
3. **Store the query stride in the dataset descriptor args.** This is wrong. Descriptors are cached and shared per
   dataset and parameter set (`dataset_descriptor_init_with_cache`), and the stride is per-search state.
4. **Test-only change: FilterTest dim 8 → 16.** This saves about 13 s each for int8/uint8, but it loses coverage
   and helps no users.

## Are results identical to the per-query path?

The plan object, the algorithm selection (AUTO is decided from `max_queries` before the loop, so it is the same in
both paths) and the kernels are the same. Only the grid's query dimension changes, from 1 to `n_queries`.

* **SINGLE_CTA (normal and persistent): bit-identical.** Each CTA handles one query independently. Random
  entry points use `block_id = 0, num_blocks = 1`, which doesn't depend on `query_id`. Seeds are indexed as
  `seed_ptr + num_seeds * query_id` and match the old `+ num_seeds * g` offset. The visited hashmap is per query
  (`hash_size * blockIdx.y`). The filter offset is `query_id_offset + query_id`, which equals the old `set_offset(g)`.
  The query values are the same elements, either the caller's buffer or an exact element copy.
* **MULTI_KERNEL: same per-query computation.** `random_pickup` seeds depend only on `global_team_index`. A query
  that converges early sits through the remaining iterations of its batch with no parents. Its itopk is re-sorted
  together with invalid child slots, so it doesn't change. This is the behaviour aligned dims have always had.
  Bit-identity with the per-query path is expected but isn't formally guaranteed, because ties in the extra
  `_find_topk` passes could in principle be reordered.
* **MULTI_CTA: not bit-identical, by design.** Random entry points use `block_id = cta_id + num_cta_per_query *
  query_id` and `num_blocks = num_cta_per_query * num_queries`, so they depend on the batch shape. Unaligned dims
  now get the same results as aligned dims always have (and as every dim did before #1846). This is a different
  random sample of the same quality, not a correctness change. The tests check recall and distance thresholds, not
  bitwise equality with the fallback.

## Memory

* Dense queries: **less** memory than before. The old unaligned path allocated `n_queries x stride` for the whole
  query set from the default device resource, plus a memset. The new path allocates nothing.
* Padded queries: `min(max_queries, n_queries) x dim x sizeof(T)` from the workspace resource, allocated once and
  reused per batch. `max_queries` defaults to `min(n_queries, maxGridSize.y)`. In the iterative build `n_queries`
  is at most the 8192-row chunk, so the buffer is 4.5 MB for float dim 137 and 32 MB for float dim 1024. That is no
  larger than the per-batch buffers the plans already take from the same resource (hashmaps, result buffers).

## Tests

Measure (wall time, `ctest -R` or a single process, before/after):

* `NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST`: profiled excess about 54 s (about 45 s after the index-reuse proposal).
  Unaligned float dims are 1, 3, 7, 17, 102 and 137.
* `NEIGHBORS_ANN_CAGRA_UINT8_UINT32_TEST`: about 50 s. Unaligned dims also include 8 and 10.
* `NEIGHBORS_ANN_CAGRA_INT8_UINT32_TEST`: about 51 s, about 16 s of it from dim 8.
* `NEIGHBORS_ANN_CAGRA_HALF_UINT32_TEST`: about 30 s (about 23 s after the index-reuse proposal).
* `NEIGHBORS_TIERED_INDEX_TEST` (dim 29), `NEIGHBORS_ANN_CAGRA_TEST_BUGS` and `NEIGHBORS_ANN_CAGRA_MERGE_TEST`.
  These are secondary.
* Controls, expected unchanged: `NEIGHBORS_ANN_CAGRA_FILTER_UDF_TEST` (dim 16), `NEIGHBORS_DYNAMIC_BATCHING_TEST`
  (dim 32) and `NEIGHBORS_ANN_CAGRA_BBQ_UINT32_TEST`.

Correctness coverage of the changed path (all must pass; they check recall and distances against brute force):

* **AnnCagraTest** (dims 1, 3, 7, 17, 137, ... × SINGLE_CTA / MULTI_CTA / MULTI_KERNEL / AUTO, plus
  `max_queries`). Dense unaligned queries are now batched.
* **AnnCagraTest with ITERATIVE_CAGRA_SEARCH builds** at unaligned dims. These cover the padded-query pack path,
  because the iterative build passes slices of the padded dataset.
* **AnnCagraAddNodesTest.** `add_nodes` passes dense `[batch, dim]` queries, so these are now batched.
* **AnnCagraFilterTest** (dims 1, 8, 17, 102; MULTI_KERNEL with `n_queries=100`). This covers the filter
  `query_id_offset` across batches. For int8/uint8, dim 8 is unaligned.
* **AnnCagraIndexMergeTest** and FilteredMerge (searches of merged indices at unaligned dims).
* **NEIGHBORS_ANN_CAGRA_TEST_BUGS:** `bug_iterative_cagra_build`, `bug_nan_queries_multi_kernel`,
  `bug_extreme_inputs_oob`, `bug_graph_smaller_than_dataset`.
* **`test_iterative_cagra_q.cu`** (in the FLOAT executable). The VPQ iterative build passes padded reconstructed
  queries. Its dims are aligned for float, so it is mainly a regression check.
* **Recommended once:** `compute-sanitizer --tool memcheck` over a few unaligned cases, to confirm empirically that
  the kernels make no misaligned query reads. For example:
  `NEIGHBORS_ANN_CAGRA_UINT8_UINT32_TEST --gtest_filter='*FilterTest*'` and
  `NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST --gtest_filter='AnnCagraTest*'` limited to dims 1 and 17.

No test covers padded queries with `persistent = true` (no gtest sets `persistent`).

## Risks

* **Misaligned reads (main risk).** The argument above rests on the query reads being scalar in both
  `setup_workspace` implementations. Any future vectorized query load would have to handle an arbitrary pitch,
  which multi-partition search already requires. The memcheck run above validates this.
* **MULTI_CTA results at unaligned dims change** (different random entry points). Recall is unaffected on average.
  Tests near their recall threshold at unaligned dims could move slightly either way.
* **Non-device-accessible queries.** Dense unaligned queries in pageable host memory used to be silently copied to
  the device by the padded copy. They now get the same "must be device-accessible" error that aligned dims already
  got. The API takes a `device_matrix_view`, and the C API checks for device-compatible memory.
* **User-set `max_queries`** is now honoured for unaligned dims (batches instead of single queries). Larger batches
  mean the per-batch plan buffers are actually used at their planned size, but the plan already allocates them for
  `max_queries` anyway.

## Verification done

* Compile-checked on an overlay copy of `HEAD`'s `cpp/src` with the edited header. The generated TUs
  `cagra_search_inst_data_u8.cu` and `cagra_search_inst_data_f.cu` were built with their exact
  `compile_commands.json` flags (`-Werror`, `-Werror=all-warnings`), the overlay `-I` ahead of `cpp/src`, a single
  `-arch=sm_89`, and `nice -n 19`. `nvcc -M` confirmed that `cagra.cuh`, `cagra_search.cuh` and `cagra_build.cuh`
  came from the overlay and that no header came from the working tree's `cpp/src`.
* `git apply --check` of `change.patch` against the working tree passed.
* Verified on a GPU: the CAGRA, tiered-index, dynamic-batching, MG and HNSW tests pass, and `compute-sanitizer --tool memcheck` reports no errors on 145 unaligned-dimension cases (see `pr.md`).
