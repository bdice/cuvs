### NN-descent spends most of each iteration on host thread and OpenMP overhead for small and medium graphs

**Problem**

`GNND::build` (`cpp/src/neighbors/detail/nn_descent.cuh`, both the dense and the BBQ overload) starts a new `std::thread` on every
iteration to run `update_graph` + `sample_graph`, and joins it after enqueuing the iteration's `add_reverse_edges` and `local_join`.
Every host step in the loop is a default-size `#pragma omp parallel for` over the graph rows.

For the graph sizes used by CAGRA/HNSW-ACE partition builds, the HNSW GPU hierarchy and the tests (150–10 k rows), this costs much more than
the GPU work it overlaps. nsys traces on an 18-core / 36-thread host (RTX 6000 Ada):

| | per iteration | GPU work per iteration | `pthread_join` total |
|---|---|---|---|
| `NEIGHBORS_ANN_NN_DESCENT_TEST` (2–4 k rows) | 2.3–3.1 ms | 0.6–1.1 ms | main thread blocked 1.1–1.7 ms per iteration; ~2.5 k `pthread_create` per group |
| `NEIGHBORS_ANN_CAGRA_INT8_UINT32_TEST` | ~1.8 ms join alone | `local_join` 2.0 s total | **20.6 s** |
| `NEIGHBORS_ANN_CAGRA_BBQ_UINT32_TEST` | 3.2–5.6 ms | 0.3–0.4 ms | 6.2 s |
| `NEIGHBORS_ANN_HNSW_ACE_FLOAT_UINT32_TEST` | — | — | 5.8 s |

The NN-descent loop is 57 % of `NEIGHBORS_ANN_NN_DESCENT_TEST`, 64 % of the BBQ test, and the largest bucket of the HNSW-ACE tests. Users
building many small graphs (ACE partitions, upper HNSW layers, batched all-neighbors) pay the same overhead.

**Why it is slow.** Every per-iteration `std::thread` is a new OpenMP root thread, so it brings up a second OpenMP team next to the calling
thread's team. That puts 2 × 36 threads on 36 hardware threads, and the idle workers of one team keep spinning while the other team works
(libomp's default `KMP_BLOCKTIME` is 200 ms). Join waits are bimodal (≈ 1 ms or 4–12 ms), which fits scheduling stalls at team barriers,
not compute. The regions themselves are small: 0.1–2 µs per row on one core, so a 4,000-row region is 0.4–7 ms of single-core work.

The helper thread also makes errors fatal. If `local_join` throws (for example the `__CUDA_ARCH__ < 700` check or a launch error) while the
thread is joinable, `std::thread`'s destructor calls `std::terminate`.

**Proposed fix**

Drop the per-iteration thread. Enqueue the GPU work first (it is all asynchronous), then run `update_and_sample` on the calling thread.
The host and device halves of an iteration touch disjoint buffers, so the overlap and all ordering are unchanged, and errors propagate.

The host loops themselves are real work that scales with cores: with the current code, `OMP_NUM_THREADS=8` is about as fast as 36
threads, `4` is 60 % slower and `1` is 4.8× slower on the NN-descent tests (see the measurements in the PR), so the OpenMP team size of
the row loops should stay as is.

**Not in scope.** The ~9 pinned allocations per `GNND` construction already go through raft's pinned memory resource. Callers can pool them
with `raft::resource::set_pinned_memory_resource`.
