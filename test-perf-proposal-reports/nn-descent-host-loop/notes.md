# nn-descent-host-loop: cheaper per-iteration host work in `GNND::build`

Patch: `change.patch` (one file, `cpp/src/neighbors/detail/nn_descent.cuh`, +4/−5; originally +42/−18, see the scope change). It applies to `HEAD` and to the current working tree
(`git apply --check` passes). Compile-checked with `-Werror` (see "Compile check" below). Committed as d3bbb9a8 (part 1 only, see below); the NN-descent, all-neighbors, CAGRA, HNSW-ACE and BBQ tests pass on a GPU.

## Scope change after the experiment (final patch)

The `OMP_NUM_THREADS` / `KMP_BLOCKTIME` experiment below was run on the unpatched build (`NEIGHBORS_ANN_NN_DESCENT_TEST`, filter
`AnnNNDescentTest/AnnNNDescentTestI8_U32.AnnNNDescent/2??:AnnNNDescentBbqTest/*`, idle 36-thread host, 2 repetitions):

| setting | wall (rep 1 / rep 2) | CPU user+sys (rep 1 / rep 2) |
|---|---|---|
| `OMP_NUM_THREADS=36` (default) | 16.6 / 16.8 s | 579 / 589 s |
| `OMP_NUM_THREADS=8` | 17.2 / 17.3 s | 253 / 255 s |
| `OMP_NUM_THREADS=4` | 26.9 / 26.6 s | 186 / 184 s |
| `OMP_NUM_THREADS=1` | 82.3 / 79.4 s | 83 / 80 s |
| `KMP_BLOCKTIME=0` | 14.6 / 14.9 s | 141 / 141 s |
| `OMP_NUM_THREADS=8 KMP_BLOCKTIME=0` | 21.9 / 21.8 s | 91 / 90 s |

Reading:

* Fewer threads never helped wall time: 8 threads are 3 % slower than 36, 4 threads 60 % slower. The host loops are real compute that
  benefits from the whole team, so sizing teams by row count (part 2 below, `clamp(nrow / 256, ...)`, 15 threads at 4,000 rows) would at
  best save CPU time, not wall time. **Part 2 was dropped from the patch** (kept as `change_with_team_sizing.patch` for reference).
* `KMP_BLOCKTIME=0` (workers sleep right after a region) is 12 % faster and uses 76 % less CPU. Spinning idle workers, i.e. the two
  teams competing for the cores, are what cost time. Removing the per-iteration `std::thread` (part 1) removes the second team, which is
  the code-level fix for that. A passive OpenMP wait policy for the CI test environment is evaluated separately in
  `../ci-openmp-passive-wait/`.

`change.patch` is now part 1 only: `+4/−5` in `cpp/src/neighbors/detail/nn_descent.cuh`. The sections below describe the original
two-part proposal; everything about part 1 (synchronization, ordering, exceptions) still applies.

## What changes

1. **No thread per iteration.** Both `GNND::build` overloads (dense, and BBQ) used to start a `std::thread` running `update_and_sample(it)`
   (`update_graph` + `sample_graph(false)`) before enqueuing the iteration's GPU work, and joined it afterwards (old lines 2754/2789 and
   2900/2917). Now the main thread first enqueues the GPU work (`add_reverse_edges` ×2 + `local_join`, all asynchronous) and then calls
   `update_and_sample(it > 0)` itself. The host work still overlaps the GPU work.
2. **OpenMP teams sized by row count.** Every row loop in the file now has `num_threads(omp_threads_for_rows(n))`:
   `shrink_graph_removing_duplicates`, `sample_graph_new`, `init_random_graph`, `sample_graph`, `update_graph`, `sort_lists`, the distance
   slice and graph copy-out in both `build` overloads, and the three copy-out loops in the `detail::build` wrappers. That is 14 regions
   (old lines 2077, 2167, 2203, 2239, 2282, 2302, 2814, 2850, 2936, 2970, 3048, 3080, 3105).
3. Nothing else changes: no API, data-layout or algorithm changes, and no change to the pinned buffers (see "Pinned buffers" below).

## Why

From the nsys traces (`../NEIGHBORS_ANN_NN_DESCENT_TEST.md`, `../NEIGHBORS_ANN_CAGRA_INT8_UINT32_TEST.md`, `../NEIGHBORS_ANN_CAGRA_BBQ_UINT32_TEST.md`,
`../NEIGHBORS_ANN_HNSW_ACE_*.md`):

* An iteration takes 2.3–5.6 ms of wall time, of which only 0.3–1.1 ms is GPU work. The main thread waits 1.1–3.8 ms in `pthread_join`, and the
  waits are bimodal.
* Total `pthread_join` wait: 20.6 s in CAGRA int8 (against 2.0 s of `local_join`), 6.2 s in CAGRA BBQ, and 5.8 s in HNSW-ACE float. The
  NN_DESCENT test makes about 2.5 k `pthread_create` calls per group.

The mechanism:

* The OpenMP runtime is LLVM libomp. The conda env's `libgomp.so.1` is a symlink to `libomp.so`, so `KMP_*` variables apply.
* Each per-iteration `std::thread` is a new OpenMP root thread, so its parallel regions run on a second team. That team sits next to the main
  thread's team, giving 2 × 36 threads on 36 hardware threads. Idle workers of one team spin for `KMP_BLOCKTIME` (200 ms by default) while the
  other team works, and a team barrier stalls whenever one of its threads is descheduled. Under `ctest -j8` this gets worse.
* The work per region is small: 2–5 k rows at 0.1–2 µs per row (measured below).

Running the host half on the main thread leaves one team, which stays hot between regions. Sizing that team by `nrow` keeps small graphs from
waking all cores.

## Synchronization and ordering argument

One iteration (`it`) of the loop, after the patch:

| step | thread | what | buffers |
|---|---|---|---|
| 1 | main | `raft::copy` ×3 + `sync_stream` | `graph_.h_list_sizes_new/old` → `d_list_sizes_new_/old_`; `graph_.h_graph_old` → `h_graph_old_` |
| 2 | main enqueues, GPU runs | `fill`, `add_rev_edges_kernel`, D2H (×2); `fill`, `local_join_kernel_*` | **reads** `graph_.h_graph_new` (pinned, zero-copy), `h_graph_old_`, `h_rev_graph_new_/old_`, `d_list_sizes_*_`, data, norms; **writes** `dists_buffer_` (scratch), `h_rev_graph_new_/old_`, `graph_buffer_`, `dists_buffer_`, `d_locks_` |
| 3 | main (was: helper thread) | `update_and_sample(it > 0)` | **reads** `graph_host_buffer_`, `dists_host_buffer_` (D2H'd and synced in step 4 of `it-1`); **writes** `graph_.h_graph` (= output graph), `graph_.h_dists`, `graph_.h_graph_old`, `graph_.h_list_sizes_old`, `graph_.h_list_sizes_new`, `update_counter_` |
| 4 | main | `break` if converged; else D2H `graph_buffer_`/`dists_buffer_` → host buffers, `sync_stream`, `sample_graph_new` | writes `graph_host_buffer_` (mark_old), `graph_.h_graph_new`, `graph_.h_list_sizes_new`, bloom filter |

* **No shared buffers.** The buffers in steps 2 and 3 do not intersect, in either version. `sample_graph(false)` does not touch
  `h_graph_new` (only `sample_new == true` does), and the kernels read the device copies of the list sizes, not the host ones.
* **Same ordering edges.**
  * 1 → 3: the `sync_stream` before step 2. Before the patch this was the thread start; now it is program order on one thread.
  * 3 → 4: was `join()`, now program order.
  * GPU part of 2 → 4: the `sync_stream` in step 4, unchanged.
  * 2 ∥ 3: still true. The enqueue sequence never blocks the host. `raft::matrix::fill` is a `linalg::map` kernel launch, `raft::copy` is a
    `cudaMemcpyAsync` into pinned memory, and `kernel_virtual_arch` is `cudaFuncGetAttributes`. So step 3 runs while the GPU works.

  The only timing difference is that step 3 starts after about 7 async enqueues (≈ 0.1 ms) instead of right after step 1. Before the patch the
  main thread spent that time idle in `join()`.
* **Termination path unchanged.** On `update_counter_ == -1` the loop breaks with this iteration's `local_join` still in flight, exactly as
  before. The post-loop `update_graph` uses the same host buffers, and the `sync_stream` after it waits for the GPU.
* **Exceptions propagate.** Before the patch, an exception from `local_join` destroyed a joinable `std::thread`, which calls `std::terminate`.
  Such exceptions include `THROW("NN_DESCENT cannot be run for __CUDA_ARCH__ < 700")`, `RAFT_CUDA_TRY(cudaPeekAtLastError())`, and the BBQ
  `RAFT_EXPECTS` checks. An exception inside the helper thread also terminated the process. Both now propagate normally.
* **Determinism.** The host output does not depend on the team size.
  * Every host loop is independent per row and writes only that row.
  * `update_counter_` is an atomic sum.
  * Each row's bloom-filter region is `512 × num_sets_per_list` bits. That is word-aligned, so rows never share a `std::vector<bool>` word.
  * `shrink_graph_removing_duplicates` seeds its random fill from the row index and uses a thread-local `seen_bits` that is reset per row.

  So, for the same `local_join` outputs, the graph is bit-identical to before. Existing nondeterminism, unchanged by this patch:
  * `insert_to_global_graph` inserts into other rows' lists under per-segment spin locks. The order depends on block scheduling, so ties and
    duplicates can resolve differently from run to run. A before/after comparison therefore has to compare recall, not exact graphs.
  * `BloomFilter::clear()` (`nn_descent_gnnd.hpp:117`) is a `parallel for` over `std::vector<bool>` bits. Chunk boundaries can share a
    64-bit word, so the read-modify-write can race and leave a stale bit set. It only runs when a `GNND` object is reused (all-neighbors
    batched build, once per cluster). Not touched here. A possible follow-up is to clear per row with `std::fill` on word-aligned ranges, which
    is race-free and runs at memset speed.

## OpenMP sizing rule

```
MIN_ROWS_PER_OMP_THREAD = 256
threads = clamp(nrow / 256, 1, omp_get_max_threads())   // cuvs::core::omp::get_max_threads()
```

| nrow | 150 (HNSW upper layer) | 1,000 | 2,000 | 4,000 | 5,000 (ACE partition) | ≥ 9,216 |
|---|---|---|---|---|---|---|
| threads (36-thread host) | 1 | 3 | 7 | 15 | 19 | 36 (unchanged) |

**Calibration.** `bench.cpp` in this directory replicates the loops. It was run single-threaded, pinned to one core with `nice -n 19` on the
i9-10980XE, with nrow = 4000:

| loop | µs per row (node_degree 96 / 32) |
|---|---|
| `update_graph`, all candidates inserted | 1.82 / 1.53 |
| `update_graph`, about half inserted | 1.57 / 0.96 |
| `update_graph`, all rejected (early exit) | 0.09 / 0.09 |
| `sample_graph(false)` | 0.43 / 0.16 |
| `sample_graph_new` | 0.47 / 0.48 |

* **Work per thread.** 256 rows give each thread about 25–470 µs of work per region. That is well above the fork/join cost of a hot team
  (a few µs), so the team stays efficient.
* **Large inputs.** They keep the whole team.
* **Why not 4096 rows per thread?** The `nrow / 4096` rule suggested in the hotspot notes would run 4000-row builds on one thread. Then
  `sample_graph_new` alone would take about 1.9 ms on the critical path, and `update` + `sample` about 2–9 ms per iteration. That is slower
  than today, so the threshold is 256, not 4096.
* **Still bounded by the environment.** `OMP_NUM_THREADS` still caps the team, since the clamp's upper bound is `omp_get_max_threads()`.
* **Nested regions.** Inside an enclosing parallel region without nesting enabled, the regions run on 1 thread, the same as before for the
  main-thread regions.

## Pinned buffers (not changed)

`GnndGraph` and `GNND` allocate 9 pinned buffers per `GNND` construction:
* `GnndGraph`: `h_graph_new`, `h_list_sizes_new`, `h_graph_old`, `h_list_sizes_old`.
* `GNND`: `graph_host_buffer_`, `dists_host_buffer_`, `h_rev_graph_new_`, `h_graph_old_`, `h_rev_graph_old_`.

This happens once per build, not per iteration. The all-neighbors batched build reuses one `GNND` across clusters. The allocations already go
through `raft::resource::get_pinned_memory_resource_ref(res)`, so a caller, or the test fixtures, can install a pooled pinned resource with
`raft::resource::set_pinned_memory_resource`. Pooling inside `GNND` would duplicate that mechanism.

Value from the traces: ≈ 2 s for NN_DESCENT (2 %), and ≈ 0.4–0.55 s each for BBQ and HNSW-ACE float. Better as a separate change.

## Experiment to run first (unpatched build, no code changes)

This tests whether OpenMP fork/join and two-team oversubscription dominate the iteration, rather than real host compute.

```bash
B=/home/coder/cuvs/cpp/build/latest/gtests/NEIGHBORS_ANN_NN_DESCENT_TEST
# 88 int8 cases at n_rows=4000 (dims 31-1024, gd 32/64, all metrics, host/device) + all BBQ cases; ~25-30 s per run
F='AnnNNDescentTest/AnnNNDescentTestI8_U32.AnnNNDescent/2??:AnnNNDescentBbqTest/*'
for v in "OMP_NUM_THREADS=36" "OMP_NUM_THREADS=8" "OMP_NUM_THREADS=4" "OMP_NUM_THREADS=1" \
         "KMP_BLOCKTIME=0" "OMP_NUM_THREADS=8 KMP_BLOCKTIME=0"; do
  for rep in 1 2; do
    echo "== $v (rep $rep)"
    ( time env $v $B --gtest_filter="$F" --gtest_brief=1 ) 2>&1 | grep -E "ms total|real|user|sys"
  done
done
```

Optionally repeat with `NEIGHBORS_ANN_CAGRA_INT8_UINT32_TEST --gtest_filter='*MultiPartition*'`, which has the 20.6 s `pthread_join` profile.
Run on an otherwise idle host, and alternate the order if the times are noisy.

How to read the results. In the unpatched code, `OMP_NUM_THREADS` sizes both the main team and each helper thread's team.

* **Confirms the diagnosis:** `OMP_NUM_THREADS=8` or `4` is ≥ 15 % faster in wall time than `36`, user+sys CPU time drops several-fold, and/or
  `KMP_BLOCKTIME=0` is faster. Overhead and oversubscription then dominate. The patch should do at least as well as the best variant, because
  it also removes the thread spawn and the second team.
* **Refutes it:** `8` and `4` are no faster, or slower, than `36`. The host work is then real compute that benefits from all cores, and the
  patch's gain would come only from removing the thread. If so, consider lowering `MIN_ROWS_PER_OMP_THREAD` (for example to 128) before
  measuring.
* **`OMP_NUM_THREADS=1`:** expected to be clearly slower. The microbenchmark predicts about 4–11 ms of host compute per iteration at 4000 rows
  and gd 64. If it is not slower, the per-row costs above overstate reality, and a larger threshold would be safe.

## Measurements after the patch

Rebuild libcuvs. Every TU that includes `nn_descent.cuh` recompiles: `nn_descent_inst_*`, `nn_descent_gnnd_inst`, `all_neighbors`, CAGRA
build, HNSW and the clustering TUs. Then:

1. Rerun the same filter with no environment overrides. Expect about 0.8–1.3 ms per iteration instead of 2.3–3.1 ms, and wall time at or
   below the best variant from the experiment.
2. Run `ctest -R` (one at a time, and also under `-j8` as in CI) on the executables listed below, before and after. All tests must pass, and
   logged recalls should stay in the same range.
3. Optional: under `nsys profile -t cuda,osrt`, there should be no `pthread_create` or `pthread_join` inside the iteration loop.

## Test executables that exercise NN-descent

| executable | how |
|---|---|
| `NEIGHBORS_ANN_NN_DESCENT_TEST` | direct: F32, I8, DistEpi (reachability epilogue), BBQ; UI8 is `DISABLED_` |
| `NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST`, `_HALF_`, `_INT8_`, `_UINT8_` | `graph_build_algo::NN_DESCENT` cases, MultiPartition, IndexMerge |
| `NEIGHBORS_ANN_CAGRA_BBQ_UINT32_TEST` | dense + BBQ NN-descent builds (exactly 20 iterations each) |
| `NEIGHBORS_ANN_HNSW_ACE_FLOAT_UINT32_TEST`, `_HALF_`, `_INT8_`, `_UINT8_` | ACE partition sub-builds (~5 k rows) + GPU-hierarchy upper layers (`hnsw.hpp:486`, ~150 rows) |
| `NEIGHBORS_ALL_NEIGHBORS_TEST` | `knn_build_algo::NN_DESCENT`, batched (`GNND` reused across clusters), mutual-reachability epilogue |
| `NEIGHBORS_ANN_CAGRA_FILTER_UDF_TEST` | `nn_descent_params` graph build (`test_filter_udf.cu:127,206`) |
| `NEIGHBORS_ANN_CAGRA_MERGE_TEST` | `graph_build_algo::NN_DESCENT` (`test_merge_fastener.cu:1082`) |
| `NEIGHBORS_ANN_CAGRA_TEST_BUGS` | `bug_issue_93_reproducer.cu:74` |

Not affected as configured: `PREPROCESSING_TEST` (spectral embedding uses brute-force all-neighbors) and `CLUSTER_TEST`.

## Risks

* **Host contention.** On a busy host, a team smaller than the core count gives up some peak throughput for small graphs. On an idle host, a
  4000-row build with 15 threads has an estimated ~0.07 ms more critical-path time per iteration (`sample_graph_new`) than an ideal 36-thread
  team. That is small next to the ~1.5–2 ms per iteration being removed. Large graphs (≥ 9,216 rows on 36 threads) are unchanged.
* **Callers inside a non-nested OpenMP region.** If `GNND::build` is called from inside an active OpenMP parallel region with nesting
  disabled, `update_graph` and `sample_graph` previously got a full team on the fresh helper thread. They now run single-threaded, like
  `sample_graph_new` already did. In-tree, the only such caller is the multi-GPU all-neighbors batch build, and it enables nesting
  (`all_neighbors_batched.cuh:508`), so it is unaffected.
* **Host work starts slightly later.** It begins about 0.1 ms after the iteration's enqueue sequence, when it used to start right away. This
  only matters when the host work is longer than the GPU work, and then the removed thread spawn and second team more than offset it.
* **Verification.** The tests above pass on a GPU (RTX 6000 Ada), and so does the full `ctest` suite.

## Compile check

Two TUs were built from their exact `compile_commands.json` commands, with these changes:
* sources taken from a copy of `cpp/src` under `$TMPDIR`;
* `-I.../cpp/src` pointed at that copy;
* every `--generate-code` replaced by `-arch=sm_89`;
* `-o` sent to `$TMPDIR`;
* run under `nice -n 19`.

The TUs:
* `cpp/src/neighbors/nn_descent_gnnd_inst.cu` instantiates `GNND<const float,int>`, both `build` overloads, and the reachability epilogue.
* `build/latest/src/neighbors/nn_descent_inst_data_f_index_u32.cu` instantiates the `detail::build` wrappers, including BBQ.

Both compile cleanly with `-Werror -Werror=all-warnings`. Before compiling, `nvcc -M` confirmed that both TUs picked up the edited
`neighbors/detail/nn_descent.cuh` and none of the repo's `cpp/src` headers. `clang-format` 20.1.8 with `cpp/.clang-format` reports no changes.
