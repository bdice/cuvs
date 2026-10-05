# NEIGHBORS_ANN_SCANN_TEST: where the time goes and how to make it faster

Source: 3 nsys runs, one per fixture group (`-t cuda,nvtx,osrt`, RTX 6000 Ada). Tests: `ann_scann/test_float_int64_t.cu`
→ `ann_scann.cuh`. This is a **small test** (10 s under `ctest -j8`). Each case builds one ScaNN index from 4096 rows
and checks code validity and reconstruction error; there is no search. Like IVF-PQ, it is k-means- and launch-bound.

## Summary

| metric | value |
|---|---|
| cases / groups | 105 / 3: TEST_P `build`, `build_host_input`, `build_host_input_overlap` × the same 35 params (`test_float_int64_t.cu:12-18`) |
| gtest time (nsys) | 13.1 s (3.3 / 4.7 / 5.1). nsys wall time is 11.4–12.8 s per group, mostly nsys finalising 225 k-kernel traces. |
| process start-up | ~0.47 s before the first case. Inside case /0: CUDA context 0.11 s plus ~0.16 s of cuBLAS/cuSOLVER init and 21 module loads. |
| GPU busy | **20–23 %** of the CUDA span. ~225 k launches per group (≈ 6.4 k per case), mean kernel 3.2 µs, 113 k pool alloc/free pairs per group. |
| biggest cases | `dim = 2048`: 0.42–0.65 s each (6 cases, 3.4 s, 26 %). `dim = 1024`: 0.22–0.39 s each (6 cases, 2.0 s, 15 %). |

## Where the time goes (per-case time by phase: 12.85 s)

Phases are split at kernel and API markers: `rngKernel` (start of the case), the AVQ cuSOLVER kernels, the first
`cudaMemcpy2DAsync` of `train_pq_subspaces`, `process_and_fill_codes_subspaces_kernel`, and `reconstruct_vectors_kernel`.

| phase | build | host_input + overlap | total | % |
|---|---|---|---|---|
| PQ codebook training (`pq::build` → `train_pq_subspaces`: one balanced k-means per subspace, `cpp/src/preprocessing/quantize/detail/pq.cuh:130-147`) | 1.76 | 2.41 + 2.49 | 6.65 | **52 %** |
| data generation + coarse balanced k-means (32 leaves, 24 iterations, hierarchical: ~180 EM passes per case) + predict | 0.91 | 1.45 + 1.63 | 3.99 | 31 % |
| AVQ: 32 per-leaf Cholesky solves, with a `sync_stream` after each (`scann_avq.cuh:614-628`) | 0.44 | 0.63 + 0.68 | 1.75 | 14 % |
| encode, copies, test checks | 0.09 | 0.17 + 0.19 | 0.46 | 4 % |

* **PQ training scales with the subspace count** (`dim / pq_dim`): 1138 per group, 256 in one `dim = 2048` case. Each
  subspace is ~12 EM passes, ~116 launches, ~1.4 ms: the launch-bound `build_clusters` loop of IVF-PQ (its ideas A/E).
* **Host-input groups are 1.4–1.8 s slower with identical GPU work** (0.71–0.73 s). Host-input-specific cost is only
  ~0.35 s per group (140 pinned alloc/free pairs 0.22 s, 0.66 GiB H2D 0.12 s). The rest is uniformly slower API calls
  (`cudaLaunchKernelExC` mean 3.9 → 5.3 µs), most likely from CPU contention during profiling (see Uncertainties).
* **`overlap` never overlaps dataset batches**: 4096 rows means one batch (`kReasonableMaxBatchSize = 65536`,
  `scann_build.cuh:200-201`). Only AVQ's per-cluster prefetch uses the second stream. Its times match `build_host_input`.

## Ideas (ranked by estimated savings; % of the 13.1 s profiled)

1. **Host-input variants on a representative subset** *(REDUCES coverage)*. Add a second fixture alias for the two
   `TEST_BUILD_HOST_INPUT*` macros (`ann_scann.cuh:401-411`, `test_float_int64_t.cu:13-14`). Instantiate it with
   `defaults() + bf16() + bf16_avq() + avq() + soar()` + dim 6/4-bit + dim 512/8-bit: 7 instead of 35 cases each.
   The host paths (`sample_rows`, batch loader, AVQ gather) do not depend on PQ settings.
   **5.3 s (40 %)** at device-group case times, 7.9 s (60 %) at measured host times; ≈ 5 s of the 10 s ctest run.
   Effort S, risk low. Optional *(ADDS coverage)*: one overlap case with > 65,536 rows, ~+0.3–0.5 s (estimated).
2. **Fewer k-means iterations in the test config** *(keeps code paths)*: `kmeans_n_iters` 24 → 10 and `pq_train_iters`
   10 → 5 in `scann_inputs` (`ann_scann.cuh:31-36`). **3.5–4.5 s (27–35 %)**, ~1.5 s after idea 1. Effort trivial.
   Risk medium: re-validate the (loose) reconstruction thresholds (`ann_scann.cuh:254-269`) on the GPU.
3. **Library: batch the per-subspace k-means** in `train_pq_subspaces` (`pq.cuh:130-147`; IVF-PQ idea A) *(KEEPS
   coverage)*. Up to 60–70 % of PQ training: **≈ 4 s (30 %)**. Effort L, risk medium (codebooks change numerically).
   It also helps every `pq::build`/VPQ user.
4. **Drop `dim = 2048`** from `big_dims_all_pq_bits()` (`ann_scann.cuh:343,350`) *(REDUCES coverage)*. From 512 to
   2048 only the subspace count grows; the kernel set is the same. **3.4 s (26 %)**, or 0.9 s (7 %) after idea 1.
5. **Library: balanced k-means per-iteration overhead** (IVF-PQ idea E: `adjust_centers` D2H sync, norms recomputed,
   ~6 alloc/free pairs per iteration) *(KEEPS coverage)*. k-means is 83 % of the time; 10–20 % of it is **1.3–2.6 s**. Effort M.
6. **Library: batch the AVQ solves** (`potrfBatched`/`potrsBatched` instead of 32 solves + syncs, `scann_avq.cuh:614-628`)
   *(KEEPS coverage)*. **≈ 1 s (8 %)**, estimated; also helps real builds (`n_leaves = 1000` by default). Effort M.

Bottom line: idea 1, or 1+4, about halves this 10 s test with little coverage loss. Ideas 3, 5 and 6 are worth doing
for ScaNN and `pq::build` users, not for CI time.

## Uncertainties

* Phase boundaries come from kernel and API markers. The export has no cuvs NVTX ranges, so boundaries are good to ~1 ms per case.
* **The host-group slowdown is attributed to background nsys export jobs, but this is inferred, not measured.** The profiling script exports the previous group while the next one runs. For this reason the conservative idea-1 figure uses device-group case times.
* nsys CUDA tracing inflates launch-bound phases. The shares should transfer; the conversion to ctest seconds is approximate (`-j8` GPU sharing, a single process).
* Ideas 2, 3, 5 and 6 were not validated on the GPU (no runs or builds were allowed). Iteration-count savings assume that time scales with EM passes.
