### [TEST] Two CAGRA bug reproducers spend ~13 s on inputs their bugs do not need

**Problem**

In `NEIGHBORS_ANN_CAGRA_BUGS_TEST`, two reproducers account for most of the run time when the JIT cache is warm
(~13 of ~15 s on an RTX 6000 Ada), and most of that time does not reach the code they guard.

* **`AnnCagraBugMultiCTACrash`** (`cpp/tests/neighbors/ann_cagra/bug_multi_cta_crash.cu`, #438) takes ~10 s. Its
  multi-CTA search takes 6 ms. The other 8.9 s is a NN-descent build (plus `optimize`) over 1,183,514 × 100 half rows.
  The bug is in the random seed selection: when every query-to-node distance is +inf, the seed index was left
  uninitialized and then used to read the graph. The queries are `upper_bound<half>()` (+inf), so every distance is
  +inf whatever the graph is. The built graph plays no part in the bug.
* **`cagra_extreme_inputs_oob_test`** (`bug_extreme_inputs_oob.cu`, #337/#460, #565) takes 3.3–3.8 s. Most of that is
  IVF-PQ training plus 25 batches of IVF-PQ search and device `refine` over 100k rows of `N(0, 1e20)` data. The
  overflow it relies on happens for any row count. It also accepts *any* `std::exception`, so a build that failed for an
  unrelated reason would still pass.

**Proposal**

This is a test-only change. The bug conditions and the search/build parameters stay the same:

* MultiCTA: keep the 1.18M × 100 half dataset, the +inf queries, the search params and the separate search
  `raft::resources`. Replace `cagra::build` + `update_dataset` with the `index(res, metric, dataset, graph)`
  constructor, fed a random graph of the original degree (32) from `raft::random::uniformInt`. The search plan
  (`max_iterations`, hash sizes) depends only on the dataset size and graph degree, so it is unchanged. Also check that
  every returned id is a dataset index or the invalid id, as `bug_nan_queries_multi_kernel.cu` already does.
* Extreme inputs: use 10k rows instead of 100k, and 100 IVF lists instead of the default 1024. This keeps ~100 rows
  per list, ~50 k-means points per cluster and ~2000 candidates per query, as before. Catch `raft::logic_error`
  only, and require the message from the prune check in `graph_core.cuh` ("...invalid or duplicated neighbor
  nodes..."). Fail if the build returns.

The expected saving is about 12 s per run of this executable. The only coverage lost is the incidental 1.18M-row
NN-descent build.
