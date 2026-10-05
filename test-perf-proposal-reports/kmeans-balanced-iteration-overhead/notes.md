# kmeans-balanced-iteration-overhead (option "E" of T2)

Patch: `change.patch` (one file: `cpp/src/cluster/detail/kmeans_balanced.cuh`, +529/-167 lines).
Checks: `git apply --check` passes against `test-perf-proposals`. Committed as 7c16a24e; the full `ctest` suite passes on a GPU, and IVF-PQ / IVF-Flat mean recall is unchanged within run-to-run noise (see `pr.md`).

## Problem

One balanced k-means EM iteration costs 133–170 µs of wall time but only 31–55 µs of GPU time.
IVF-PQ codebook training runs this loop once per subspace (`ivf_pq_build.cuh:349-415`) or once per list
(`:458-512`). It does about 22–26 iterations each time, on inputs of at most 64 k rows × `pq_len` columns.
The IVF_PQ test executable alone runs 4.24 M iterations.

Per iteration, the original code does the following (line numbers are in the unpatched header):

* **`adjust_centers`.** It copies the cluster sizes to the host, synchronizes, and sorts them (`:810-819`). It
  does this for both donor modes, although the Random donor never uses the sorted pairs. When a pair exists, it
  also:
  * allocates two receiver/donor buffers that the Random donor doesn't use (`:846-847`);
  * allocates two `device_scalar`s and memsets each (`:850-855`);
  * reads `update_count` back with a second blocking D2H copy plus sync (`:872`).
* **`predict`.** When the caller passes no norms (`build_clusters` from IVF-PQ, VPQ, PQ preprocessing and
  ScaNN), it recomputes the dataset norms on every iteration (`:542-547`, `:586-609`). It also allocates
  `cur_dataset_norm`, plus the three `predict_core` buffers (`:93-101`), on every call.
* **`calc_centers_and_sizes`.** It allocates the cub histogram's temp storage on every call (`:268`).

## What changed (all in `kmeans_balanced.cuh`)

1. **Workspace reuse.**
   * `predict` is split into a thin wrapper and `predict_with_workspace`. The new function takes the minibatch
     size and a `predict_workspace<MathT>` holding:
     * `cur_dataset` and `cur_dataset_norm`;
     * the `predict_core` buffers;
     * the `predict_core_half` scratch, with `centers_ready` reset on each call.
   * `predict_core` takes its three buffers by reference. The inner-product path reuses `norm_or_dist`, which
     resolves the old `TODO: pass buffer`.
   * `calc_centers_and_sizes` gets an overload that takes the histogram workspace. The old signature forwards
     to it, so the public `helpers::calc_centers_and_sizes` is unchanged.
   * `adjust_centers` (SizeSorted) takes one `device_uvector<IdxT>` scratch buffer for
     `[update_count, receivers, donors]`.
   * `balancing_em_iters` owns all of these buffers, so iterations after the first allocate nothing.
2. **Norms computed once per fit.**
   * Change: `balancing_em_iters` computes the norms once, with the same `compute_norm` call that `predict`
     made, and passes them as `dataset_norm`.
   * Conditions: the caller passed no norms, the metric is L2, L2Sqrt or cosine, the type pair is not
     half→float, and the dataset fits in one predict minibatch.
   * Helpers `map_minibatch` and `compute_minibatch_norm` are shared by `predict` and this precompute, so the
     two cannot diverge.
3. **Minibatch size computed once.** `calc_minibatch_size` is evaluated once per fit, before any scratch buffer
   is allocated, and passed to `predict_with_workspace`. The new scratch buffers live across iterations, so they
   would otherwise shrink the free workspace that `predict` sees.
4. **Random donor without per-iteration host round trips.** This is the main gain.
   * The kernel now also records, per call, whether any cluster is underfull or overfull (`any_unbalanced`). It
     takes its seed from a small device table, `seeds[n]`, where n is the number of earlier calls that recorded
     `any_unbalanced`. Each call writes a 3-element record: `[search_count, update_count, any_unbalanced]`.
   * `adjust_centers_random` only launches that kernel: no host copy, sort, memset, allocation or sync.
   * `balancing_em_iters` reads the records back only when the loop is about to end. It then applies, in order,
     the same `balancing_counter`/`n_iters` updates and the same `i_primes` advances as before, and continues if
     iterations were added. So each fit has 1–3 syncs instead of about 1.2 per iteration.
   * The process-wide `i_primes` and the prime table move into `donor_seed_state<…>()` and `next_donor_seed()`.
     They keep the same per-instantiation state, shared by both donor modes.
5. The SizeSorted path keeps its host sort and per-iteration sync, which it needs for the pairs. It gets the
   reused scratch and adds a `cudaPeekAtLastError` after its launch.

## Why results are identical (same inputs, same `i_primes` at entry)

* **Norms.** The dataset never changes inside the loop. The precompute runs only in the single-minibatch case,
  where `predict` called `compute_norm(norm, dataset, dim, n_rows, …)` with exactly the same arguments
  (`raft::linalg::norm` picks its kernel from `(N, D)`). The half→float path (`predict_core_half` computes its
  own norms) and the multi-minibatch case are left untouched.
* **Minibatch size.** Before the patch, nothing allocated in `balancing_em_iters` outlived an iteration. So the
  free workspace seen by `predict`'s `calc_minibatch_size` equals the value at loop entry, which is where it is
  now computed. The minibatch partition is therefore unchanged.
* **Reused buffers.** No scratch buffer is read before it is written: the KVP output is filled or
  `init_out_buffer`, GEMM uses `beta = 0`, and the cub temp storage is scratch. Only addresses change, and every
  address still comes from an rmm allocation with the same alignment.
* **Random donor: the kernel runs exactly when the old code ran it.**
  * Before, the host-side size-sorted pairing found a pair, which advanced `i_primes` and launched the kernel,
    iff:
    * n_clusters ≥ 2;
    * the largest cluster is non-empty;
    * `!(min ≥ lower) || !(max ≤ upper)`.
  * The integer→MathT conversion is monotone, so the last condition is equivalent to "some cluster has
    `!(size ≥ lower) || !(size ≤ upper)`", which the kernel evaluates per cluster.
  * The sizes are the histogram of the labels produced by the preceding M step, so n_rows > 0 implies a
    non-empty cluster. An out-of-range label would already have made `reduce_rows_by_key` write out of bounds.
  * The kernel only modifies underfull clusters. So whenever the old code skipped the kernel, the new kernel
    changes nothing.
* **Random donor: same seed sequence.** Every call that would have advanced `i_primes` uses `seeds[n]`, the
  (n+1)-th `next_donor_seed` result from the entry value. That is the prime the old code computed at that point.
* **Random donor: same loop length and final state.** `n_iters` only matters when the loop is about to end. The
  pending outcomes are applied in call order before that test, so the loop runs the same iterations. The final
  `i_primes` is the same as well.
* **Host model check.** A host-only model of the bookkeeping matched the original logic exactly (same seeds used,
  `n_iters`, final `i_primes`) in 100,800 randomized scenarios, 29,664 of which added iterations
  (`bookkeeping_model.cpp` in this directory; `g++ -std=c++20 bookkeeping_model.cpp && ./a.out`).
* **Caveat (pre-existing, unchanged).** `reduce_rows_by_key` sums with float `atomicAdd`, and the Random kernel
  hands out donor candidates through an `atomicAdd` counter. The baseline can therefore differ in the last bits,
  or in which donor is picked, from run to run. "Identical" means the same computation and the same seed
  sequence: whenever the baseline is reproducible on an input, the patched build gives the same output.

## Expected effect (estimates, not measured)

Per EM iteration on the IVF-PQ codebook path (float, L2, Random, 256 clusters, one minibatch):

| | before | after |
|---|---|---|
| blocking D2H + sync | ≈1.2 (1, or 2 when a pair exists) | ≈0.1 (1–3 per fit) |
| `cudaMallocAsync`/`cudaFreeAsync` pairs | ≈5.8 | 0 (after the first iteration) |
| kernel launches | ≈10.5 | ≈10.3 (norm kernel gone, balancing kernel now always launched) |
| memsets | ≈1.4 | 1.0 (from `reduce_rows_by_key`) |
| host sort of the cluster sizes | every iteration | none |

* **Per-iteration time.** Host launches can now run ahead of the GPU. Per-iteration wall time should approach
  the host's launch-and-bookkeeping time instead of host time + GPU time + two round trips. A guess is about
  147 µs → 70–100 µs, i.e. 30–50 % less per iteration.
* **IVF_PQ.** PQ training is about 78 % of the executable, so savings could be 15–30 % of IVF_PQ wall time.
  That is at or above the 50–100 c estimated for E, because that estimate didn't include removing the
  `update_count` sync.
* **SizeSorted users.** These keep one sync per iteration and save about 6 alloc/free pairs and one norm kernel
  per iteration:
  * coarse IVF training via `build_hierarchical` (IVF-Flat, IVF-SQ, RaBitQ, ScaNN, IVF-PQ centers);
  * PQ preprocessing, VPQ and ScaNN PQ.

## Tests

* **Correctness (must pass):**
  * `CLUSTER_TEST --gtest_filter='KmeansBalancedTests*'` (float/int32/int64, cosine, half, int8 paths);
  * `PREPROCESSING_TEST --gtest_filter='ProductQuantizationTests*'` (balanced `build_clusters`, SizeSorted,
    no norms → norm hoist);
  * all `NEIGHBORS_ANN_IVF_PQ_*` (the only in-tree user of the Random donor; recall thresholds).
* **Not covered by the in-tree k-means tests.** None of them sets `donor_selection = Random`. Adding a
  small Random case to `get_kmeans_balanced_inputs` would cover the new async path directly. It is not in the
  patch because it couldn't be validated on a GPU here.
* **Bitwise check (optional).**
  1. Build the same IVF-PQ index twice in the baseline. If its `pq_centers` are bitwise reproducible on that
     input, they should be bitwise equal after the patch too.
  2. Do the same for `cluster::kmeans::fit` with `donor_selection = Random`.
* **Executables to time (before/after).**
  * Largest win (IVF-PQ codebooks, Random donor):
    * `NEIGHBORS_ANN_IVF_PQ_TEST`, `_FLOAT_TEST`, `_INT8_TEST`, `_UINT8_TEST`, `_INT8_HOST_INPUT_TEST`,
      `_BIG_DIMS_SEARCH_TEST`, `_BIG_DIMS_HOST_INPUT_TEST`, `_BIG_DIMS_EXTEND_SERIALIZE_TEST`;
    * CAGRA ×4 with the IVF-PQ graph build: `NEIGHBORS_ANN_CAGRA_{FLOAT,HALF,INT8,UINT8}_UINT32_TEST`;
    * `NEIGHBORS_DYNAMIC_BATCHING_TEST`, `NEIGHBORS_TIERED_INDEX_TEST`, `NEIGHBORS_ALL_NEIGHBORS_TEST`,
      `NEIGHBORS_MG_TEST`, `NEIGHBORS_ANN_HNSW_ACE_*` (all build IVF-PQ or CAGRA indexes).
  * Smaller win (SizeSorted coarse/PQ training):
    * `NEIGHBORS_ANN_IVF_FLAT_TEST` and `_{FLOAT,HALF,INT8,UINT8}_INT64_TEST` (HALF exercises the reused
      `predict_core_half` scratch);
    * `NEIGHBORS_ANN_IVF_SQ_TEST`, `NEIGHBORS_ANN_IVF_RABITQ_TEST`, `NEIGHBORS_ANN_SCANN_TEST`;
    * `CLUSTER_TEST`, `PREPROCESSING_TEST`.

## Compile check

* **Method.** Each TU was compiled with its exact `compile_commands.json` command, with these changes: source
  and `cpp/src` include dir → the overlay, `-o` → `$TMPDIR`, gencodes → `-arch=sm_89`, `-t=1`, `nice -n 19`.
  `-Werror` was kept, and `nvcc -M` confirmed that only overlay headers were used.
* **TUs.** All of these compile clean:
  * `kmeans_balanced_predict_float.cu`: the `build_clusters` instantiation used by IVF-PQ, both donor paths;
  * `kmeans_balanced_fit_half.cu`: the native half path;
  * `kmeans_balanced_fit_int8.cu`: T ≠ MathT;
  * `kmeans_balanced_predict_half.cu`;
  * `preprocessing/quantize/pq.cu`;
  * `ivf_pq_build_extend_inst_data_f_index_i64.cu`;
  * `ivf_flat_build_extend_inst_data_h_index_i64.cu`.
* **Formatting.** The header is clang-format-20 clean with `cpp/.clang-format`.

## Risks

* **`static i_primes` and threads.** For the Random donor, `i_primes` is now read at loop entry and written at
  the end, so two threads fitting concurrently with the same template arguments can lose an advance. The old
  code was already a data race, and its seeds then depended on timing. Single-threaded sequences are unchanged.
* **Memory.**
  * Scratch now lives for the whole fit. Peak usage is the old `predict` peak plus the histogram temp storage
    and a few `IdxT`s.
  * The precomputed norm buffer replaces `cur_dataset_norm` and has the same size, n_rows ≤ minibatch.
* **Host runs ahead.** With the Random donor, the host can queue a whole fit (about 10 launches × 25 iterations)
  before its first sync. Each fit now also ends with a sync. That sync makes `train_per_cluster`'s host reads of
  managed memory safer than before.
* **Error reporting.** A failure in a Random-donor kernel now surfaces at the end-of-fit sync, or at the
  `cudaPeekAtLastError` right after the launch, rather than in the same iteration.
* **TF32 norms.** When `top_1_nn` picks the TF32 norm policy, it ignores the precomputed norms. That costs one
  unused norm kernel per fit, versus one per iteration before.
* **Out of scope.**
  * Batching all subspaces into one EM loop (option A).
  * Moving the SizeSorted sort and pairing to the device (cub radix sort plus a pairing kernel), which would
    remove its per-iteration sync too.
