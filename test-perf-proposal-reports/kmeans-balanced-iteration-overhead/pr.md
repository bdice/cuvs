# Reduce per-iteration host overhead in balanced k-means

On small inputs, balanced k-means EM iterations were dominated by host overhead. This matters most for IVF-PQ
codebook training, which runs one fit per PQ subspace. This PR removes that overhead without changing results
for the same inputs and RNG state.

## Changes (`cpp/src/cluster/detail/kmeans_balanced.cuh`)

* **Reuse scratch buffers across iterations.**
  * `predict` is split into a wrapper and `predict_with_workspace`, which takes a `predict_workspace` and a
    precomputed minibatch size.
  * `predict_core` and `calc_centers_and_sizes` take their buffers from the caller.
  * `adjust_centers` uses a single scratch vector.
  * `balancing_em_iters` computes the minibatch size once, before allocating anything, so `predict` still uses
    the same minibatch partition.
* **Compute the dataset norms once per fit**, when the caller passes none and the dataset fits in one minibatch.
  The call is the same `compute_norm` that `predict` made on every iteration.
* **Random donor: no per-iteration host round trip.**
  * The kernel records whether any cluster is underfull or overfull. That is exactly when the old host-side
    size-sorted pairing found a pair and advanced `i_primes`.
  * It takes its seed from a precomputed table, indexed by the number of earlier such calls.
  * The per-call outcomes are read back only when the loop is about to end. They are then applied in order to
    the balancing pull-back counter and to `i_primes`.

  This removes the cluster-size D2H copy, the stream sync and the host sort, as well as the unused
  receiver/donor buffers, the two `device_scalar`s and the second sync.
* SizeSorted keeps its host sort, which it needs, and benefits from the buffer reuse.

## Why results don't change

* **Norms and minibatch size.** The norms come from the same call on the same unchanged data. The minibatch
  size is the value `predict` computed on every iteration before, because nothing allocated in the loop
  outlived an iteration.
* **Reused buffers.** None is read before it is written.
* **Random donor.**
  * The per-cluster check `!(size >= lower) || !(size <= upper)` is equivalent to the old `min`/`max` pair
    check. The sizes are the label histogram, so the largest cluster is non-empty when `n_rows > 0`.
  * The kernel only changes underfull clusters, so in the calls the old code skipped, it changes nothing.
  * Seeds, iteration count and the final `i_primes` match the old logic. A host model of the bookkeeping agreed
    with the original in 100,800 randomized scenarios, including pull-backs.
* **Unchanged nondeterminism.** The float atomics in `reduce_rows_by_key` and the donor-candidate counter can
  already make two runs differ. They are untouched.

## Testing

* The full `ctest` suite passes (`ctest -j8`; 59/60, the one failure is `cuvs_c_verify_install_headers`, which needs
  installed headers and fails the same way without this change).
* Recall is unchanged within run-to-run noise: over the 994 recall checks of `NEIGHBORS_ANN_IVF_PQ_TEST`, the mean
  recall is 0.8710 / 0.8701 in two runs before and 0.8705 / 0.8709 in two runs after; `NEIGHBORS_ANN_IVF_FLAT_TEST`
  (776 checks) 0.8678 / 0.8684 before, 0.8672 / 0.8673 after.

## Measurements

Single-process wall time of each test executable on an RTX 6000 Ada (48 GB) with a 36-core host, otherwise idle (no ctest parallelism, no MPS). Old and new builds were run alternately, 2 or more repetitions each with the order reversed between repetitions; mean (min–max).

The library changes on this branch were measured as a chain: each executable was run against six builds of `libcuvs.so` (before any of them, then after each one, cumulatively), alternating, 2 repetitions per build. For this PR, "before" is the build just before this change and "after" adds only this change; the test binaries are identical. Executables affected only by this change were measured separately in the same way.

| executable | tests before | tests after | before | after | change |
|---|---|---|---|---|---|
| `NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST` | 1417 | 1417 | 30.5 s (30.4–30.6) | 24.3 s (24.0–24.5) | -20.4% |
| `NEIGHBORS_ANN_CAGRA_HALF_UINT32_TEST` | 883 | 883 | 17.3 s (17.0–17.5) | 14.3 s (14.3–14.4) | -17.0% |
| `NEIGHBORS_ANN_CAGRA_INT8_UINT32_TEST` | 943 | 943 | 20.9 s (20.6–21.3) | 17.3 s (17.1–17.6) | -17.1% |
| `NEIGHBORS_ANN_CAGRA_UINT8_UINT32_TEST` | 943 | 943 | 21.7 s (21.4–21.9) | 18.0 s (17.8–18.3) | -16.7% |
| `NEIGHBORS_ANN_HNSW_ACE_FLOAT_UINT32_TEST` | 27 | 27 | 4.1 s (4.1–4.1) | 4.0 s (4.0–4.1) | -2.1% |
| `NEIGHBORS_ANN_HNSW_ACE_HALF_UINT32_TEST` | 25 | 25 | 6.2 s (5.8–6.6) | 6.0 s (5.8–6.2) | -2.9% |
| `NEIGHBORS_ANN_HNSW_ACE_INT8_UINT32_TEST` | 25 | 25 | 4.0 s (3.9–4.0) | 3.9 s (3.9–4.0) | -1.3% |
| `NEIGHBORS_ANN_HNSW_ACE_UINT8_UINT32_TEST` | 25 | 25 | 4.2 s (3.8–4.7) | 4.0 s (3.9–4.1) | -5.4% |
| `NEIGHBORS_ANN_CAGRA_BBQ_UINT32_TEST` | 135 | 135 | 1.9 s (1.9–2.0) | 1.9 s (1.8–1.9) | -4.4% |
| `NEIGHBORS_ANN_IVF_PQ_TEST` | 1065 | 1065 | 258.1 s (257.7–258.5) | 167.9 s (167.7–168.0) | -35.0% |
| `NEIGHBORS_ANN_IVF_FLAT_TEST` | 390 | 390 | 145.5 s (145.0–145.9) | 141.6 s (140.3–142.8) | -2.7% |
| `NEIGHBORS_ANN_IVF_SQ_TEST` | 134 | 134 | 21.7 s (21.6–21.9) | 21.8 s (21.6–22.0) | +0.4% |
| `NEIGHBORS_ANN_IVF_RABITQ_TEST` | 230 | 230 | 10.6 s (10.6–10.7) | 10.2 s (10.1–10.3) | -4.4% |
| `NEIGHBORS_ANN_SCANN_TEST` | 105 | 105 | 7.1 s (7.1–7.1) | 5.5 s (5.5–5.5) | -21.6% |
| `CLUSTER_TEST` | 157 | 157 | 17.4 s (17.3–17.4) | 17.3 s (17.2–17.4) | -0.2% |
| `PREPROCESSING_TEST` | 226 | 226 | 4.4 s (4.4–4.4) | 4.2 s (4.2–4.2) | -4.7% |
| `NEIGHBORS_DYNAMIC_BATCHING_TEST` | 281 | 281 | 30.5 s (30.4–30.6) | 30.6 s (30.4–30.7) | +0.2% |
| `NEIGHBORS_TIERED_INDEX_TEST` | 72 | 72 | 4.0 s (4.0–4.0) | 3.1 s (3.1–3.1) | -22.3% |
| `NEIGHBORS_ALL_NEIGHBORS_TEST` | 228 | 228 | 20.7 s (20.4–20.9) | 16.7 s (16.6–16.8) | -19.1% |
| `NEIGHBORS_MG_TEST` | 15 | 15 | 14.5 s (14.5–14.5) | 14.4 s (14.3–14.5) | -0.8% |
| **total** | | | 645.2 s | 527.1 s | -18.3% |

All tests passed in every run.

Closes #`<issue>`
