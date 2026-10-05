# Balanced k-means: host overhead dominates each EM iteration on small inputs

## Summary

`cluster::kmeans::detail::balancing_em_iters` (`cpp/src/cluster/detail/kmeans_balanced.cuh`) spends most of each
iteration on host overhead rather than GPU work when the input is small. That is the normal case for IVF-PQ
codebook training, which runs one balanced k-means per PQ subspace, or per list with `PER_CLUSTER`. Each run
does ~20–26 iterations on at most `256 × max_train_points_per_pq_code` rows of `pq_len` columns.

On the IVF-PQ gtests (`dim = 6144`, `pq_dim = 3072`), nsys shows 147 µs of wall time per EM iteration against
31 µs of GPU time. The executable runs 4.24 M such iterations, so PQ training is ~78 % of its runtime.

## Where the time goes (per iteration)

* **`adjust_centers`** copies the cluster sizes to the host, synchronizes the stream, and sorts them. It does
  this even with `balanced_donor_selection::Random`, the mode IVF-PQ uses, which never uses the sorted pairs.
  When rebalancing triggers, it also:
  * allocates receiver/donor buffers that the Random path never reads;
  * allocates and memsets two `device_scalar`s;
  * synchronizes again to read `update_count`.
* **`predict`** recomputes the dataset row norms on every iteration when the caller passes none.
  `build_clusters` is called that way from IVF-PQ, VPQ/PQ preprocessing and ScaNN. The dataset doesn't change
  across iterations.
* **Scratch buffers** are allocated and freed on every call: `cur_dataset_norm`, the three `predict_core`
  buffers, and the cub histogram storage in `calc_centers_and_sizes`. That is ~6 `cudaMallocAsync`/`cudaFreeAsync`
  pairs per iteration.

Per iteration that adds up to ~10.5 launches, ~6 alloc/free pairs, 1–2 blocking D2H copies with syncs, and a
host sort. Because of the syncs, the host cannot run ahead of the GPU, so launch overhead and GPU time add up
instead of overlapping.

## Proposal

Without changing results for the same inputs and RNG state:

1. Allocate the EM scratch buffers once per fit and reuse them across iterations.
2. Compute the dataset norms once per fit, when the dataset fits in one predict minibatch, with the same call
   `predict` makes.
3. For the Random donor:
   * have the kernel record whether any cluster is underfull or overfull (the condition under which the
     host-side pairing found a pair);
   * have it pick its seed from a small precomputed table indexed by the number of earlier such calls;
   * read the per-call outcomes back only when the loop is about to end.

   The seeds, the centers, the number of iterations (including balancing pull-backs) and the process-wide
   `i_primes` stay the same.

Out of scope: batching all subspaces into one EM loop, and removing the SizeSorted path's per-iteration sync,
which needs a device-side sort.
