# cagra-bug-reproducer-inputs

Test-only change to `NEIGHBORS_ANN_CAGRA_BUGS_TEST`. Two bug reproducers get cheaper inputs that still meet their bug
conditions:

* `AnnCagraBugMultiCTACrash` (#438) no longer builds a graph over 1,183,514 × 100 half rows. Its index gets a random
  graph of the original degree.
* `cagra_extreme_inputs_oob_test` (#337/#460, #565) uses 10k rows instead of 100k, with the IVF list count scaled down
  to match. It now asserts the exception type and message instead of accepting any `std::exception`.

The iterative-build reproducer (#1818) is not touched. Background: `../NEIGHBORS_ANN_CAGRA_BUGS_TEST.md`
ideas 1 and 3.

## What changed

Both files are formatted with clang-format 20.1.8 (repo `.clang-format`; already clean, no diff).

### `cpp/tests/neighbors/ann_cagra/bug_multi_cta_crash.cu`

* `run()` `:31-38`: `cagra::build` (NN-descent with intermediate degree 48, then `optimize` to degree 32) and
  `cagra::update_dataset` are gone. The index is now built directly:
  `cagra::device_padded_index<half> cagra_index(res, metric, build_padded_->view, make_const_mdspan(graph->view()))`.
  This is the dataset+graph constructor (`cpp/include/cuvs/neighbors/cagra.hpp:628-651`). It stores the same padded
  dataset view as before and a non-owning view of the graph. `ann_vamana.cuh:220-221` uses it the same way.
* `SetUp()` `:74, :80-85`: new member `graph` [n_samples, 32] `uint32_t`. It is filled with
  `raft::random::uniformInt` in `[0, n_samples)`, using the same `RngState` **after** `InitDataset`, so the dataset
  values are unchanged. It is reset in `TearDown()` (`:98`). New constant `graph_degree = 32` (`:116`). The unused
  `index_params` and the `<utility>` include are removed.
* Unchanged: `n_samples = 1183514`, `n_dim = 100`, `half`, L2Expanded, the `upper_bound<half>()` (+inf) queries,
  `n_queries = 30`, `k = 10`, all search params (MULTI_CTA, itopk 32, thread_block_size 256, search_width 1,
  max_iterations 0), and the separate `raft::resources res_search`.
* New output check (`:56-68`). The neighbors are copied on `res_search`'s stream. Each id must be `< n_samples` or the
  invalid id `0xFFFFFFFF`. Before, the test only checked that nothing crashed. Async-pool allocation makes an OOB read
  less likely to fault, and the check catches a garbage seed index that leaks into the output without one. This is the
  check `bug_nan_queries_multi_kernel.cu` uses for the same class of bug.

### `cpp/tests/neighbors/ann_cagra/bug_extreme_inputs_oob.cu`

* `n_samples` 100000 → 10000 (`:72`). `n_dim` stays 200, the data stays `N(0, 1e20)` with seed 1234.
* `gb_params.build_params.n_lists = 100` (`:31-32`). The default is 1024. See below for why it scales with the rows.
  All other IVF-PQ build and search params stay at their defaults. `refinement_rate = 2`, degrees 64/128 are unchanged.
* The padded copy is made before the `try` (`:37-38`), so unrelated failures, such as OOM, are not swallowed.
* The `try` (`:42-50`) now ends with `FAIL()` when `cagra::build` returns. It catches only `raft::logic_error` (the
  type `RAFT_EXPECTS` throws) and asserts that `what()` contains `"invalid or duplicated neighbor nodes"`. That text
  exists only in `RAFT_EXPECTS` at `cpp/src/neighbors/detail/cagra/graph_core.cuh:1530-1534` (`prune_graph_gpu`,
  added by #565). Any other exception type now escapes and fails the test with its message. A `raft::logic_error`
  with another message (e.g. from k-means or an argument check) fails the `EXPECT_NE` and prints `what()`.
  `<raft/core/error.hpp>` and `<string>` are now included.

## Why each reproducer still triggers its bug condition

### #438 MULTI_CTA illegal address (`bug_multi_cta_crash.cu`)

The bug (PR #438, commit eff2cc5c) was in `compute_distance_to_random_nodes`. Today this is
`compute_distance_to_random_nodes_jit` in `cpp/src/neighbors/detail/cagra/jit_lto_kernels/device_common_jit.cuh:36-101`.
For each seed slot, it samples `num_distilation` random node ids and keeps the best one with
`if (norm2 < best_norm2_team_local)`, starting from `best_norm2 = upper_bound<float>() = +inf`. If **every** sampled
distance is +inf, nothing is ever closer. `best_index_team_local` then kept its uninitialized value, which was inserted
into the visited hash and the result buffer. `pickup_next_parent` then chose that index as a parent, and
`compute_distance_to_child_nodes` read `knn_graph[... + knn_k * parent_id]` out of bounds
(`search_multi_cta_jit.cuh:218-266`). The fix initializes the index to `upper_bound<IndexT>()` and skips it (`:63`,
`:87`).

The trigger is "every query-to-node distance is +inf". It depends only on the queries and the dataset:

* `raft::upper_bound<half>()` is `0x7c00` = +inf (`raft/util/cudart_utils.hpp`). The query is stored as `float` in
  shared memory (`query_t = float` for L2, `compute_distance_standard-impl.cuh:102-105`). Each term is
  `(inf - x)^2 = inf` for any finite dataset value x (`dist_op_l2_impl`, `dist_op_impl.cuh:14-19`). So every node's
  distance is +inf, whatever the graph.
* The seed ids are drawn as `xorshift64(...) % seed_index_limit`, where `seed_index_limit = graph_size` (= n_samples,
  unchanged).
* The graph is only read **through** a parent id taken from the result buffer, and that id comes from the seed phase
  above. With the fix, no valid parent is ever picked (`parent_indices_buffer[0] == invalid_index`, so the loop breaks).
  Without the fix, the parent is a garbage id, and the out-of-bounds read does not depend on the graph's contents.

Everything that shapes the kernel launch and the search plan is unchanged:

* Dataset size and dim, `half`, the same padded view (same row width, same descriptor).
* Graph shape [1,183,514, 32].
* itopk 32, search_width 1, thread_block_size 256, MULTI_CTA.
* Auto `max_iterations`, which comes from `dataset_size` and `graph_degree` (`search_plan.cuh:200-216`), and the
  hash sizes derived from it.
* The fresh `res_search`, which the PR notes make the bug easier to hit.

What is lost: the incidental coverage of a 1.18M-row half NN-descent build plus `optimize` (the 8.9 s), and the exact
memory layout the build left behind in the pool. #438 was intermittent and tied to whatever memory lies past the
graph. The graph allocation is the same size (151 MB) but sits at a different place in the pool. The test was already
a weak detector (async pool, no output check). The new output check makes it somewhat stronger.

### #337/#460 refine and prune OOB, #565 validation (`bug_extreme_inputs_oob.cu`)

The condition: on `N(0, 1e20)` data, every squared difference overflows float. For σ = 1e20, a component of `x - y`
stays below `sqrt(FLT_MAX) ≈ 1.8e19` with probability about 0.1. With 200 dims, the chance that a pair has no
overflowing component is about 1e-200. So every non-self L2 distance is +inf, and every squared norm is +inf, which
makes the expanded-form distances NaN or inf. None of this depends on `n_samples`.

The bug path has the same steps as before (`cagra_build.cuh` `build_from_device_matrix`, then
`build_cagra_host_graph_from_knn_params`, then `build_knn_graph<IVF-PQ>` at `:1900-2180`):

1. IVF-PQ build. `n_rows (10000) >= n_lists (100)` holds (`ivf_pq_build.cuh:1262`). The k-means trainset is
   `max(0.5·n, n_lists)` = 5,000 rows.
2. IVF-PQ search in batches of 4096 (3 batches instead of 25). `top_k = 129`, and
   `gpu_top_k = min(max(128·2, 129), n_rows) = 256`, unchanged for any `n_rows ≥ 256`. The dataset is on the device
   and `top_k != gpu_top_k`, so `async_host_processing` is false and the **device** `cuvs::neighbors::refine` runs.
   That is the #337 path: candidates with invalid ids are skipped by `build_index_kernel`
   (`ivf_flat_build.cuh:119-124`, #460).
3. `refine` scans each query's candidates with `warp_sort_filtered`, whose threshold starts at +inf. With every
   non-self distance at +inf, nothing but the self match (distance 0) is ever added. The remaining output slots stay
   at the dummy position 0, which `postprocess_neighbors` maps to the first candidate. So each refined row is
   `[self, c0, c0, …]`, or worse. `write_to_graph` drops self. The kNN row is then 128 copies of one id, or
   invalid/unset entries.
4. `optimize` runs with `guarantee_connectivity = false` (the default, so no MST step), then `prune_graph_gpu`.
   `kern_fused_prune` skips ids `>= graph_size` (#460, `graph_core.cuh:142`). It marks repeated ids as used
   (`:202-204`) and sets `d_invalid_neighbor_list` once fewer than 64 distinct valid neighbors are left (`:195-198`).
   The host check then throws at `:1530`.

Step 3 needs only all-inf distances, so the throw does not depend on `n_samples`.

**Why `n_lists` scales with `n_samples`.** With the default 1024 lists, 10k rows would leave ~10 rows per list and
~5 k-means training points per cluster. 20 probes would then reach only ~195 candidates for 256 slots. The IVF-PQ
stage would then produce invalid candidates just from a lack of data, a different and more mundane cause. 100 lists
keep the per-query statistics of the original 100k/1024 setup:

| | before (100k, 1024 lists) | after (10k, 100 lists) |
|---|---|---|
| rows per list | ~98 | ~100 |
| k-means points per cluster | ~49 | 50 |
| candidates per query (20 probes) | ~1950 | ~2000 |
| IVF-PQ candidates kept (`gpu_top_k`) | 256 | 256 |

So any invalid ids still come from the overflow, as before. Only the number of rows (queries) and lists shrinks. `pq_dim`
(96 for dim 200), `pq_bits`, `kmeans_n_iters` and n_probes are unchanged.

What is lost: 90k rows of the same computation. Nothing in the path above branches on the row count beyond the batch
count.

## Expected effect

From the profile in `NEIGHBORS_ANN_CAGRA_BUGS_TEST.md` (RTX 6000 Ada; nsys gtest time / ctest time):

| test | before | expected after | where the time went |
|---|---|---|---|
| `AnnCagraBugMultiCTACrash` | 9.82 s / 10.42 s | ~0.2–0.5 s | NN-descent 8.9 s + optimize 0.35 s removed. What remains: allocating and filling the 237 MB dataset, the padded copy, a 151 MB random graph, and the 6 ms search |
| `cagra_extreme_inputs_oob_test` | 3.30 s / 3.80 s | ~0.5–1 s | search + refine (1.95 s, per-query list allocations) scales with rows (~10×). Coarse k-means is smaller. PQ codebook training uses 5k instead of 50k points per subspace, but keeps the same number of launches (96 subspaces), so it shrinks less |

The combined saving is ~12 s: about a third of the 35.6 s cold ctest run of this executable, and most of the ~15.4 s warm
run. Peak device memory also drops: no NN-descent workspace, and 8 MB instead of 80 MB for the extreme-inputs data.
A cold JIT cache still costs the same: the multi-CTA kernel and the IVF-PQ/IVF-Flat kernels are the same as before.

## What the parent should verify on the GPU

1. **Both tests pass:**
   `./NEIGHBORS_ANN_CAGRA_BUGS_TEST --gtest_filter='AnnCagraBugMultiCTACrashReproducer*:cagra_extreme_inputs_oob_test*'`.
   For the extreme test, a pass already proves the throw came from `prune_graph_gpu` (`graph_core.cuh:1530`). The
   message is unique, and the type is checked.
   *If it fails*, the printed `what()` gives the throw site:
   * A k-means message (`kmeans_balanced.cuh`) would mean the smaller trainset changed the IVF-PQ build. Try
     `n_lists = 1024` (library default) at 10k rows, or n_samples 20–30k.
   * "did not reject" would mean refine produced a valid graph. This is not expected; see step 3 above.
2. **Profile the extreme test** (optional): nsys/NVTX should show `cagra::build_knn_graph<IVF-PQ>(10000, 200, 128)`, 3×
   `neighbors::refine(…, 256)` / `ivf_flat::fill_refinement_index`, then `cagra::graph::optimize/prune`, and nothing
   after it. The "Self-included ratio is low" warning is expected.
3. **compute-sanitizer**, which is now affordable. `compute-sanitizer --tool memcheck` on both tests should report
   0 errors. This is the original #337/#438 symptom check. Before, it would have run the 1.18M-row NN-descent build
   under the sanitizer.
4. **Trigger check for #438** (optional, temporary edit): all 300 returned ids should be `0xFFFFFFFF`, i.e. no seed
   beat +inf. Swap the `ASSERT_LT` for `ASSERT_EQ(neighbors_h[i], 0xFFFFFFFFu)` and run once.
5. **Mutation check** (strongest, optional, needs a library rebuild): revert one fix at a time and run under
   compute-sanitizer. For #438, in `device_common_jit.cuh`, leave `best_index_team_local` uninitialized (`:63`) and
   drop the `!= upper_bound` guard (`:87`). Expect an invalid read in the multi-CTA kernel, or the new `ASSERT_LT` to
   fire. Like the original, this may be intermittent. For #460, drop `graph_core.cuh:142` or
   `ivf_flat_build.cuh:121-124`. Expect memcheck errors in `kern_fused_prune` / `build_index_kernel`.
6. Timings for the PR's Measurements table: before/after gtest time of the two tests and of the whole executable
   (ctest and a single warm process).

## Not done (possible follow-ups)

* `n_dim` 200 → 32 in the extreme-inputs test. The overflow would still happen at any dim. It would cut `pq_dim` from
  96 to 32, and with it the per-subspace codebook launches (maybe another ~0.3 s). Left out to keep the reproducer
  close to the original; worth it only if the measurement shows PQ training dominating.
* Library: `fill_refinement_index` allocates one list per query (`ivf_flat_build.cuh:489-491`). That is idea 4 in the
  analysis and speeds up every device `refine`.
