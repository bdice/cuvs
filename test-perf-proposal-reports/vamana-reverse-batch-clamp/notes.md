# vamana-reverse-batch-clamp

Clamp Vamana's reverse-edge batch size to the number of dataset rows, and lower the
`NEIGHBORS_ANN_VAMANA_TEST` ctest GPU reservation from `PERCENT 100` to `PERCENT 10` (first proposed as 5; see "Measured peak" below).

Files: `cpp/src/neighbors/detail/vamana/vamana_build.cuh`, `cpp/tests/CMakeLists.txt` (see `change.patch`).

## Every use of `reverse_batchsize` / `max_reverse_batch` (line numbers are before the patch)

| Location | Use |
|---|---|
| `cpp/include/cuvs/neighbors/vamana.hpp:77` | `uint32_t reverse_batchsize = 1000000;` (public parameter, default 1e6) |
| `c/include/cuvs/neighbors/vamana.h:60`, `c/src/neighbors/vamana.cpp:39,180`, `python/cuvs/cuvs/neighbors/vamana/vamana.pxd:26`, `vamana.pyx:87,106,119,154-155` | Pass-through or default only |
| `cpp/tests/neighbors/ann_vamana.cuh:65,146,320,338,357,376` | Test parameter: `{100, 1000000}` for the deg-32 block, `1000000` for deg 64/128/256 |
| `vamana_build.cuh:145` | `int reverse_batch = params.reverse_batchsize;` (function-scope copy) |
| `vamana_build.cuh:201` | `s_coords_mem` rows = `min(maxBlocks, max(max_batchsize, reverse_batch))` (10,000 rows with the default) |
| `vamana_build.cuh:253` | `const int max_reverse_batch = params.reverse_batchsize;` |
| `vamana_build.cuh:301-302` | `reverse_list_ptr`: `max_reverse_batch` x `QueryCandidates` (32 B each) |
| `vamana_build.cuh:303-306` | `rev_ids`, `rev_dists`: `max_reverse_batch x visited_size` x 4 B each |
| `vamana_build.cuh:496-500` | `reverse_batch = params.reverse_batchsize;` then `for (rev_start = 0; rev_start < unique_dests; rev_start += reverse_batch)`, with the last batch shortened to `unique_dests - rev_start` |
| `vamana_build.cuh:502-507` | `init_query_candidate_list(reverse_list, rev_ids, rev_dists, reverse_batch, visited_size)` |
| `vamana_build.cuh:510` | `num_blocks = min(maxBlocks, reverse_batch)` (grid of all reverse kernels) |
| `vamana_build.cuh:513-550` | `reverse_batch` is the item count of `populate_reverse_list_struct`, `recompute_reverse_dists`, `RobustPruneKernel`, `SortPairsKernel`, `write_graph_edges_kernel` |
| `vamana_structs.cuh:982-1004` | `init_query_candidate_list`: entry `i` points at `ids_ptr + i * maxSize`, so the row stride is `visited_size`, **not** `max_reverse_batch` |
| `vamana_structs.cuh:1185-1215` | `populate_reverse_list_struct`: `i < reverse_batch`, reads `unique_indices[rev_start + i (+1)]` |
| `robust_prune.cuh:108` | `s_coords_mem[blockIdx.x * (dim + align_padding)]`, so it needs `gridDim.x` rows (forward grid `min(maxBlocks, step_size)`, reverse grid `min(maxBlocks, reverse_batch)`) |

No stride, offset or kernel index depends on the allocated row count (`max_reverse_batch`). Every
access is bounded by the per-launch count `reverse_batch` (or `blockIdx.x < gridDim.x` for `s_coords_mem`).

## The clamp

```cpp
const int max_reverse_batch = static_cast<int>(std::min<int64_t>(params.reverse_batchsize, N));
```

It is defined once, next to the other parameters (it replaces `int reverse_batch` at :145). It is used for:

- the `s_coords_mem` rows (:201);
- the `reverse_list_ptr` / `rev_ids` / `rev_dists` extents (:301-306), with the old :253 definition removed;
- the start value of each insert batch's reverse loop (:496, now a loop-local `int reverse_batch = max_reverse_batch;`).

The `int64_t` comparison also avoids the old `uint32_t` -> `int` narrowing for values above `INT_MAX`.

## Why results are identical

- Every reverse-list entry is a distinct destination node id. All ids are valid rows: the reverse pass writes
  `graph(queryId, j)`, so an invalid id would already be an out-of-bounds write. So `unique_dests <= N`.
- **`reverse_batchsize >= N` (e.g. the 1e6 default).** Before the patch, the first iteration sets
  `reverse_batch = unique_dests`, so there is one batch.
  - After the patch, `reverse_batch` starts at `N >= unique_dests`.
  - It becomes `unique_dests` (or already equals it when `N == unique_dests`), so there is again one batch.
  - The launches, grids, counts and `rev_start` values are identical.
- **`reverse_batchsize < N`.** `max_reverse_batch == reverse_batchsize`. The code path is byte-for-byte the same,
  and the test's `rb = 100` rows still exercise multi-batch reverse processing.
- The `QueryCandidates` pointers use a `visited_size` stride from the buffer start. Shrinking the buffer only
  removes unused trailing rows, so the data layout seen by the kernels does not change.
- **`s_coords_mem`.** It needs `gridDim.x <= min(maxBlocks, max(step_size, reverse_batch))` rows.
  - `step_size <= N` (:327) and `reverse_batch <= min(reverse_batchsize, N)`.
  - So the clamped size `min(maxBlocks, max(max_batchsize, min(rb, N)))` still covers every launch.
  - For `rb < N` the size is unchanged.
- **Other inputs.** No RNG call (`rand()` for the permutation and medoid), kernel argument or loop bound
  depends on the allocation sizes. Only buffer sizes, and therefore pool addresses, change.
- **For users.** Behaviour is unchanged for `N >= reverse_batchsize`, i.e. any dataset with at least 1e6 rows at the default.

## Device memory per test case (computed from the code; to be confirmed by measurement)

Assumptions:
- N = 1000 and 100 queries.
- `sizeof(QueryCandidates<uint32_t, float>) = 32`; `DistPair` and `Node` are `__align__(16)`, i.e. 16 B.
- `QueryCoordT` is float for float/half and 1 B for int8/uint8.
- The peak occurs inside `batched_insert_vamana`: all scratch is live there, plus the test's database and queries.
- The post-build phase (index graph + padded dataset view/copy, CAGRA recall check with 10-query batches) is
  smaller: at most about 11 MB.

The table gives the worst case over the 4 dtypes x 15 dims of each block (always float, dim 1024):

| Block (deg / visited_size / max_fraction / reverse_batchsize) | Before | After |
|---|---|---|
| 32 / 64 / 0.06 or 0.1 / 100 | 5.3-5.5 MB | unchanged |
| 32 / 64 / 0.06 or 0.1 / 1e6 | 590 MB | 9.5-9.6 MB |
| 32 / 256 / 0.06 or 0.1 / 100 | 5.7-6.1 MB | unchanged |
| 32 / 256 / 0.06 or 0.1 / 1e6 | 2.13 GB | 11.3-11.6 MB |
| 64 / 128 / 0.06 / 1e6 | 1.10 GB | 10.3 MB |
| 64 / 512 / 0.06 / 1e6 | 4.17 GB | 14.0 MB |
| 128 / 256 / 0.06 / 1e6 | 2.13 GB | 12.0 MB |
| 256 / 512 / 0.06 / 1e6 | 4.18 GB | 15.4 MB |
| 256 / 1024 / 0.06 / 1e6 | **8.27 GB (7.7 GiB)** | **20.3 MB** |

The worst case after the patch (deg 256, vs 1024, float, dim 1024) breaks down as:
- `rev_ids` + `rev_dists`: 8.2 MB
- `s_coords_mem`: 4.1 MB
- database + queries: 4.5 MB
- `d_graph`: 1.0 MB
- `topk_pq_mem`: 1.0 MB
- `visited_*`: 0.5 MB
- edge/sort scratch: about 1 MB

Before the patch the same case had:
- `rev_ids` + `rev_dists`: 8.19 GB
- `reverse_list`: 32 MB
- `s_coords_mem`: 41 MB

The process-wide RMM peak is therefore expected to drop from about **8.3e9 B to about 2.0e7 B**, with some
slack for cub/thrust temporaries. The script is at `/tmp/vrbc/mem_estimate.py` (scratch, not part of the patch).

**This must be confirmed by measurement**:
```
GTEST_CUVS_MEMORY_PEAK=1 ./gtests/NEIGHBORS_ANN_VAMANA_TEST          # whole executable
GTEST_CUVS_MEMORY_PEAK=1 ./gtests/NEIGHBORS_ANN_VAMANA_TEST \
  --gtest_filter='AnnVamanaTest/AnnVamanaTestF_U32.AnnVamana/314'   # worst case: deg 256, vs 1024, dim 1024
```

## Measured peak (GPU, `GTEST_CUVS_MEMORY_PEAK=1` + NVML, whole executable)

| build | wall | NVML process peak | RMM peak |
|---|---|---|---|
| before the clamp (`snap_s6`) | 277.5 s | 8,596 MiB | 7,890 MiB |
| after the clamp (`snap_s7`) | 277.2 s | 724 MiB | 19 MiB |

With the rule used for the other executables (`5 * ceil(peak * 1.2 / 16 GiB * 100 / 5)`, minimum 5), 724 MiB gives
**PERCENT 10**. The 5 estimated below would leave only ~95 MiB of headroom over the measured NVML peak, so the commit
uses 10. `compute-sanitizer --tool memcheck` reports no errors on four cases (rb = 100 and rb = 1e6, float and int8).

## PERCENT choice (original estimate): 100 -> 5

- Other entries in `cpp/tests/CMakeLists.txt` mostly use `PERCENT 100`. SCANN uses `PERCENT 1`, and the
  `ConfigureTest` default without GPUS/PERCENT is 30. There is no comment in the file about the convention.
- #2722 added `test_main.cpp` / `cpp/scripts/gtest_memory_usage.sh` so that PERCENT can be sized from the RMM peak.
  Its description notes that the RMM peak is about the NVML per-process peak minus about 450 MB of CUDA context.
- With 16 GB as the smallest supported GPU, `PERCENT p` reserves `p% x 16 GB`.
  - The estimated process footprint is about 20 MB (RMM) + about 450 MB (context), roughly 0.47 GB.
  - That is 2.9% of 16 GB.
  - `PERCENT 5` = 0.8 GB gives about 1.7x headroom and leaves room for growth.
  - `PERCENT 1` (SCANN-style, RMM-only) would also cover the RMM peak. It undercounts the context, so it is not proposed.
- **If the measured RMM peak is X**, use `p = ceil(100 x 1.5 x (X + 0.45 GB) / 16 GB)`. For example, X up to
  about 80 MB keeps p = 5, and X = 0.3 GB gives p = 8.

## Risks

- **Compute contention.** With `PERCENT 5`, ctest can co-schedule other GPU tests next to Vamana's ~317 s run.
  - GreedySearch is latency-bound (at most 15 blocks per launch), so it may slow down somewhat when SMs are shared.
  - Suite wall time should still drop, but this is not measured.
  - Measure `ctest -j8` wall time before and after.
- **Hidden out-of-bounds accesses.** The 1e6-row slack used to absorb any out-of-bounds access past `reverse_batch` rows.
  - Kernel indexing was checked (all accesses are bounded by `reverse_batch` or `gridDim.x`).
  - A `compute-sanitizer --tool memcheck` run on a few cases (e.g. `/314` and a deg-32 `rb=1e6` case) is still a cheap confirmation.
- **Future test cases.** Larger N or larger `visited_size` in the test would need PERCENT revisited.
  With N rows the reverse scratch is `8 x N x visited_size` bytes.
- **Unchanged pre-existing issues.**
  - `reverse_batchsize == 0` still never terminates the reverse loop.
  - `max_fraction x N < 1` still yields `max_batchsize = 0` with `step_size = 1`.
  - Both are outside this change.
  - A tighter clamp `min(rb, N, max_batchsize x degree)` is possible, but it would not change the test footprint.
- **The public header doc is not changed.** `vamana.hpp:76`, "Max batchsize of reverse edge processing", is still
  accurate. It was left alone to avoid rebuilding every TU that includes it.

## Checks done

- `vamana_build_float.cu` compiled from an overlay copy of `cpp/src` with the exact `compile_commands.json`
  command, `-arch=sm_89`, `-Werror`: OK (48 s, exit 0).
  - The only message is nvcc's `compiler-bindir` redefinition warning. It comes from the conda env's
    `NVCC_PREPEND_FLAGS=-ccbin=...` and is unrelated.
  - `nvcc -M` confirmed that the edited `vamana_build.cuh` from the overlay was used.
- `clang-format` 20.1.8 (`cpp/.clang-format`) leaves the edited header unchanged.
- `git apply --check change.patch` passes against the working tree.
- Not run: GPU tests or measurements (a timing session was running).
