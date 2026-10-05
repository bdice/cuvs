# NEIGHBORS_ANN_NN_DESCENT_TEST: where the time goes and how to make it faster

Source: 14 nsys runs, one per fixture group (`-t cuda,nvtx,osrt`, RTX 6000 Ada, i9-10980XE with 18 cores / 36 threads).
All times below are gtest time under nsys. The whole executable took **76 s** under `ctest -j8`, so "≈ real" figures
are the percentage × 76 s. Phases were reconstructed from main-thread OSRT and CUDA events. cuVS NVTX ranges are not
compiled in, and CPU sampling was unavailable.

## Summary

| metric | value |
|---|---|
| cases / groups | 916 instantiated / 14 groups. Only **580 cases run a build**: the 288 `UI8` cases are `DISABLED_` (`test_uint8_t_uint32_t.cu:16`), and 48 F32 `BitwiseHamming` cases call `GTEST_SKIP` (`ann_nn_descent.cuh:81`) |
| total gtest time (nsys) | **103.3 s**: F32 `AnnNNDescent` 48.2, I8 `AnnNNDescent` 44.4, `AnnNNDescentBbq` 6.6, `AnnNNDescentDistEpi` 3.9, UI8 0 |
| GPU busy (kernels + copies) | 19.7 s = **19 %** of the CUDA span. The test is **host-bound**. |
| kernel launches | 121 k (≈ 210 per case). There are 19.8 k NN-descent iterations (≈ 34 per case). |
| biggest groups | F32 shard 4/4 (host input, L2/IP/L1) 19.9 s (anomalous, see below), F32 3/4 12.0 s, I8 shards 10.4–11.7 s each, F32 1/4 and 2/4 8.7 / 7.7 s, BBQ 6.6 s, DistEpi 3.9 s |
| most expensive cases | `n_rows=4000, graph_degree=64`: 305 ms mean, 136 cases = 41.5 s (40 %) |

## Where the time goes

| phase (580 built cases, 102 s) | s | % | what it is |
|---|---|---|---|
| **NN-descent iteration loop** | **58.5** | **57** | library `GNND::build` loop, `nn_descent.cuh:2748-2797` (BBQ: `:2894-2923`). GPU kernels are only ≈ 12.4 s of this. |
| `check_unique_indices` | **18.0** | **18** | test helper `ann_utils.cuh:164-193`, which builds a `std::set` per row (≈ 0.2 µs per entry: 62 ms for 4000×64) |
| `calc_recall` | 8.0 | 8 | `ann_utils.cuh:228-280`: O(rows·k²), run as two passes (31 ms for 4000×64) |
| build setup | 5.8 | 6 | GNND constructor (≈ 9 `cuMemAllocHost` + 9 `cuMemFreeHost` per case, 2.0 s of API time; the first pinned allocation costs 4–5 ms), `init_random_graph`, `sample_graph(true)`, H2D copy, norms |
| host stall at next case start | 5.6 | 5 | no CUDA or OS calls between SetUp's `rngKernel` launch and `naive_knn`. Mean 10 ms after a case that built (up to 50 ms), 0.3 ms after a skipped case. |
| final update/sort/shrink | 3.4 | 3 | `nn_descent.cuh:2799-2850` (host, OpenMP) |
| `naive_knn` reference | ≈ 2.2 | 2 | `naive_distance_kernel` takes 2.05 s of GPU time. **Not worth optimizing.** |

* **The loop is host-bound.** One iteration takes 2.3–3.1 ms of wall time (5.5 ms in g003). GPU work is only 0.6–1.1 ms of it:
  * `local_join_kernel_{simt,wmma}`: 0.25–0.8 ms
  * 2× `add_rev_edges_kernel`: 0.1 ms
  * copies: 0.25 ms

  Each iteration on the host:
  1. The main thread launches the kernels (0.12 ms).
  2. It blocks in `pthread_join` (`:2789`) for **1.1–1.7 ms** while a helper thread runs `update_graph` + `sample_graph` (`:2733-2746`). The GPU finishes after about 0.5 ms.
  3. It then does a D2H copy, a serial `sample_graph_new` (`:2796`), 3 H2D copies and a sync (`:2749-2752`), taking 1.0–1.6 ms.

  Every host step is a default-size (36-thread) OpenMP region over only 2–4 k rows (`:2167`, `:2239`, `:2282`). The helper is a **new `std::thread` every iteration** (`:2754`): 2.5 k `pthread_create` per group. It runs its OpenMP regions in a second thread pool, so 72 OpenMP threads share 36 hardware threads. Join waits are bimodal (≈ 1 ms or 4–12 ms), which suggests fork/join and scheduling overhead dominates, not compute (see Uncertainties).
* **The iteration count drives the cost.** The fixtures set `max_iterations = 100` (`ann_nn_descent.cuh:112,269,390`, `ann_nn_descent_bbq.cuh:125`); the library default is 20. Median iterations: `graph_degree=32`: 15 (9–53); `graph_degree=64`: **49** (16–89, up to 142 in DistEpi, which builds twice). `graph_degree=64` cases are ~70 % of the time. Logged recall is 0.96–0.99 against `min_recall` 0.90.
* **Verification costs a quarter of the test.** `eval_neighbours` (`ann_utils.cuh:284`) runs `calc_recall` and then `check_unique_indices`, both single-threaded with no early exit. `calc_recall`'s second pass computes an index-only recall. `eval_neighbours` discards it (`:295`), and so does the other caller (`ann_cagra.cuh:1067`, which only uses `get<0,2,3>`).
* **Redundant cases.**
  * Every `host_dataset=true` case rebuilds the graph from the same data as its device twin. Only the ingest path differs (fp32→fp16 downcast for float with dim > 16, otherwise a plain copy; `nn_descent.cuh:2640-2714`). These cases cost 50.3 s.
  * `n_rows` 2000 vs 4000 take the same code paths; the 4000 cases cost 56.1 s.
  * Each case regenerates its dataset and reruns `naive_knn`, but that is cheap (≈ 2 s).
* **g003 is an outlier.** It runs the same work as g002 (host instead of device input) but takes 5.5 vs 2.5 ms per iteration, with join waits of 0.5–12 ms. This is most likely host contention during that run, and it adds ≈ 8 s to the total.
* **Not hotspots:**
  * Process start: first CUDA call at ≈ 0.5 s, `cuLibraryLoadData` ≤ 0.05 s per group. Runs g009–g012 are pure start-up (3.3 s each) because they only contain DISABLED cases; under ctest they cost nothing.
  * DistEpi (3.9 s) and BBQ (6.6 s) have the same per-iteration profile as the main fixture.
  * Unlike CAGRA, library GPU kernels are not the problem; this is the same "host-bound library loop" pattern as IVF-PQ k-means.

## Ideas (ranked by standalone estimated savings; % of the 103 s)

The ideas overlap: B roughly halves the per-iteration cost, so C saves about half as much after B. **Recommended combination
that keeps coverage: D + B ≈ 45–55 s (≈ 45–50 %, ≈ 35 s real), then C once recall is revalidated.**

**A. Trim the parameter grid** *(reduces coverage)*
* Change: `inputs` at `ann_nn_descent.cuh:466-477`.
  * Use `n_rows` {2000} only: 56.1 s (54 %).
  * Restrict `host_dataset=true` to `graph_degree=32 × dim {4, 31, 1024}` × all metrics. This keeps fp32-SIMT direct copy, fp16 downcast and odd dims: 41.7 s (40 %).
  * Both together: **72.7 s (70 %, ≈ 53 real)**. A pairwise design over dim×metric×gd would give a similar result.
* Lost coverage: the random-graph stride and rounding at `nrow=4000`, and the host ingest path for most dim/metric combinations (all from code reading).
* Effort: S. Risk: L–M.

**B. Remove the per-iteration host overhead in `GNND::build`** *(keeps coverage; library)*
* Change: call `update_and_sample(it)` inline after launching `add_reverse_edges`/`local_join` instead of creating a `std::thread` (`nn_descent.cuh:2754/2789`, `2900/2917`). The calling thread only waits in `join` anyway, and the kernels are asynchronous. Also size the OpenMP teams to the work, e.g. `num_threads(clamp(nrow/4096, 1, omp_get_max_threads()))` on the `GnndGraph` loops (`:2077, 2167, 2203, 2239, 2282, 2302, 2814, 2850`).
* Target: ≈ 1.2 ms per iteration, which is roughly GPU-bound.
* Savings: **≈ 20–35 s (20–35 %)**. This probably also removes most of the 5.6 s stall.
* Check first without code changes: run with `OMP_NUM_THREADS=1/4/8`. If that helps, a stop-gap is `set_tests_properties(... ENVIRONMENT OMP_NUM_THREADS=4)` in `cpp/tests/CMakeLists.txt:294-302`.
* The gain should be larger under `ctest -j8`, where 8 processes × 72 OpenMP threads oversubscribe 36 hardware threads.
* Effort: S–M. Risk: L–M (guard so that large-`nrow` builds keep full parallelism).

**C. Cap NN-descent iterations in the tests** *(keeps code paths, changes the configuration)*
* Change: `max_iterations` 100 → 20 (the library default) at `ann_nn_descent.cuh:112,269,390` and `ann_nn_descent_bbq.cuh:125`.
* Savings: **27.0 s (26 %)**; with 30 iterations, 17.5 s (17 %).
* The `graph_degree=32` cases still converge on their own (median 15 iterations), so the termination path stays covered.
* Effort: S. Risk: **M**. Recall for `graph_degree=64` after 20 iterations must be revalidated on the GPU, including BBQ thresholds as low as 0.15–0.35.

**D. Make the host verification helpers linear** *(keeps coverage; test, shared)*
* Change, `check_unique_indices` (`ann_utils.cuh:164-193`): replace the per-row `std::set` with a reused row buffer plus `std::sort` and an adjacent compare, or a seen-bitmap like `shrink_graph_removing_duplicates`. Saves ≈ 17 s.
* Change, `calc_recall` (`:228-280`): drop the unused index-based pass, or fold it into the first loop. Optionally match against a per-row sorted copy of the expected ids and distances. Saves ≈ 4–7 s.
* Savings: **≈ 24 s (23 %, ≈ 18 s real)**. This also speeds up every other user of `eval_neighbours`/`calc_recall` (IVF-PQ and IVF-Flat call out `calc_recall` too).
* Effort: S. Risk: L (keep the exact match semantics: same id, or distance within `CompareApprox(eps)`).

**E. Pool the pinned host buffers** *(keeps coverage; library)*
* Change: `GnndGraph`/`GNND` allocate ≈ 9 pinned buffers per build (`nn_descent.cuh:2141-2144`, `2346-2355`). Reuse them through a pooled pinned memory resource.
* Savings: ≈ 2 s (2 %). Effort: M. Risk: L.

**F. Small test items** *(mostly keep coverage)*
* DistEpi runs a full `nn_descent::build` only to get core distances (`ann_nn_descent.cuh:277`) before its `gnnd.build`: ≈ 1.5–2 s.
* BBQ uses `graph_degree=64` everywhere (53 iterations per case); C applies to it too.
* `naive_knn` could be shared between host/device twins and across `graph_degree` (top-32 is a prefix of top-64): ≈ 1 s.

**G. CI scheduling** *(no change to this executable's time)*
* `PERCENT 100` (`cpp/tests/CMakeLists.txt:300-301`) reserves the whole GPU slot for a test that is 19 % GPU-busy and uses < 100 MB of device memory.
* Lowering it lets `ctest` overlap other GPU tests with it.

## Uncertainties

* **No CPU sampling.** Attributing the join wait and post-join time to OpenMP fork/join overhead, rather than real compute, is an inference: the work is tiny (≈ 4 k rows × 32–96 entries) and the timings are bimodal. B's range depends on this, and the `OMP_NUM_THREADS` experiment settles it.
* **The 5.6 s stall** has no traced calls. SetUp's `raft::interruptible` sync polls `cudaStreamQuery` plus `yield`, which nsys does not record. That it only follows cases that built suggests leftover helper or OpenMP thread activity, but this is unproven.
* **Host contention.** g003's 2× slower iterations show how sensitive the loop is to it. Under `ctest -j8` the host-bound share is likely higher, while nsys OSRT/CUDA tracing inflates API-heavy parts (103 s summed vs 76 s real).
* **Verified on the GPU?** No. The recall impact of C and the coverage judgement in A are from code reading only.
* **Termination test.** It scales with `dataset_dim` (`nn_descent.cuh:2740-2741`), so small dims must reach (near) zero sampled updates before they stop. This may be unintended; it is out of scope for the test.
