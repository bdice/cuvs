# NEIGHBORS_ANN_VAMANA_TEST: where the time goes

Source: 16 `nsys` runs (4 fixtures F/F16/I8/U8 x 4 gtest shards), RTX 6000 Ada (142 SMs), async RMM MR.
Per-case attribution: the timeline is split at each case's first `rngKernel` launch (from `SetUp`) and
matched to the gtest case order (all 1260 cases matched).

## Summary

| Metric | Value |
|---|---|
| Cases | 1260 = 4 dtypes x 315 params (`generate_inputs`, `ann_vamana.cuh:310-388`) |
| Total gtest time (16 profiled processes) | 339.3 s (about 323 s without the 15 extra process start-ups; `ctest -j8`: 317 s) |
| GPU busy (kernel time / CUDA span) | 81% (273 s of 337 s) |
| Kernel launches | 1.18 M (about 940 per case) |
| Per dtype | F32 86.0 s, F16 85.1 s, I8 84.5 s, U8 83.8 s (the dtype makes almost no difference) |
| Biggest groups | Every shard takes 19.3-24.1 s. Shard 2/4 of each dtype is the largest (about 23.5 s). |
| Bound by | **The library Vamana build on the GPU (78%)**, using tiny, latency-bound grids |

## Where the time goes

**By parameter block** (`generate_inputs`). Every case builds a distinct index, so no index is built twice.

| Block (rows) | Cases | gtest s | % | GreedySearch s | RobustPrune s | CAGRA recall check s |
|---|---|---|---|---|---|---|
| deg 256, vs {512,1024} (l.368-385) | 120 | 147.5 | 43% | 124.7 | 17.2 | 0 (skipped: `graph_degree < 256` guard, l.186) |
| deg 32, 2x2x2x2 factorial (l.312-328) | 960 | 134.4 (118 w/o start-up) | 40% | 57.7 | 12.8 | 6.4 |
| deg 64, vs {128,512} (l.330-346) | 120 | 33.9 | 10% | 25.3 | 3.1 | 0.9 |
| deg 128, vs 256 (l.349-365) | 60 | 23.5 | 7% | 17.8 | 2.7 | 0.6 |

Mean case time: deg32/vs64 0.07 s, deg32/vs256 0.16 s, deg64/vs512 0.43 s, deg128 0.39 s, deg256 1.15-1.31 s.
Cost grows with dim: 1-17 is about 0.15 s/case, 619/1024 about 0.46 s/case. Within deg 256, dim 1024 costs
2.4 s/case and dim 1 costs 0.65 s/case.

**By phase** (all cases):

| Phase | s | % | Notes |
|---|---|---|---|
| Vamana build kernels (library `vamana::build`) | 265.4 | 78% | `GreedySearchKernel` 225.5 s (66%), `RobustPruneKernel` 35.7 s (10.5%), sorts/reverse-edge 4.2 s |
| Host gaps inside the build | ~7 | 2% | Per-batch `sync_stream` for `total_edges` and `get_n_components` (`vamana_build.cuh:434-436,474`) |
| `CheckGraph` + `vamana::serialize` (non-first cases) | 37.8 | 11% | About 30 ms/case. About 27 s is a CPU-only gap of about 9.7 ms right after each `kvikio_ofstream` opens (see idea 4) |
| Recall check (`naive_knn`, CAGRA index from graph, search, `eval_neighbours`) | 10.3 | 3% | `search_multi_cta` 7.5 s GPU (about 50 launches/case, `max_queries=10`). `naive_distance_kernel` takes 0.06 s |
| Per-process start-up inside the first case | 17.6 | 5% | About 1.05 s per process (16 here, 1 in ctest). `CUFileInit` takes 0.97 s (includes a 0.29 s `system()`) on the first device write |
| `SetUp` data generation | 0.4 | 0.1% | Negligible |

**Why the build is slow even at n_rows=1000.** `batched_insert_vamana` inserts in batches that grow
1, 2, 4, ... up to `max_fraction*N` = 60, so there are 22 batches (31 when iters=1.5). `GreedySearchKernel` runs
one warp per insert with `grid = ceil(step/4)`, which is at most **15 blocks of 128 threads on a 142-SM GPU**
(`vamana_build.cuh:331,341`). That means fewer than 10% of SMs are occupied, so each launch is limited by the
serial search of a single warp. Once the graph fills, a deg-256/vs-512 launch takes 60-90 ms (one dim-137 case:
22 launches totalling 1.03 s). With visited_size of 512 or 1024 against N=1000, each insert visits half to all of
the dataset and expands 256 neighbours per visited node. This is why the deg-256 rows are 10-20x more expensive
than the deg-32 rows. Because the GPU is "busy" but almost empty, lowering kernel time needs fewer or larger
batches, or fewer cases.

**The 8.5 GB peak for n_rows=1000 comes from a library over-allocation.** `max_reverse_batch = params.reverse_batchsize`
(`vamana_build.cuh:253`, library default 1,000,000 at `vamana.hpp:77`) sizes `rev_ids` and `rev_dists` at
`max_reverse_batch x visited_size x 4 B` each (`:303-306`). For vs=1024 that is **2 x 4.1 GB = 8.2 GB**. On top of
that come `reverse_list` (1e6 structs, about 32 MB) and `s_coords_mem` (`min(10000, 1e6) x dim x 4 B`, up to 41 MB,
`:198-202`). The rows actually used never exceed `unique_dests <= N = 1000`.

In the profile this shows up as one 100-150 ms `cudaMallocFromPoolAsync` (pool growth) in the first rb=1e6 deg-256
case of each process. `cudaMemPoolDestroy` takes 151 ms in shards that ran vs=1024 cases and 76 ms in the others.
The time cost is small (about 0.25 s/process), but the memory is why the test is declared `PERCENT 100`
(`cpp/tests/CMakeLists.txt:309`, giving `RESOURCE_GROUPS 1,gpus:100`). As a result it holds the whole GPU under
ctest for about 317 s while using about 15 blocks per kernel.

## Ideas (ranked by estimated savings; base about 323 s single-process)

1. **Trim the dim sweep for the deg 64/128/256 blocks.** Change the dim lists at `ann_vamana.cuh:332,351,370` to
   R = {1, 8, 64, 137, 384, 619, 1024}.
   - **Savings:** -92 s (about 28%). Applied to the deg-256 block only it saves -66 s. A more aggressive {8, 137, 1024} for deg-256 saves -112 s.
   - **Effort:** trivial. **Risk:** low.
   - **REDUCES coverage** (dims for the large degrees), but R keeps every dim-dependent branch:
     - odd-dim scalar L2, dim <64 SEQ, 64-127 ILP2, >=128 ILP4
     - prune smem cache and 256-thread prune (>=128), fp16 query smem (>=512), half multi-warp prune (>=960)
     - both codebook paths (64/float, 384/int8)
   - It also keeps every (degree, visited_size) sort instantiation, including the 1024-wide sort, and all 4 dtypes. The full 15-dim sweep with the recall check stays on the deg-32 rows.
2. **Use `max_fraction` 0.1 instead of 0.06 for the deg 64/128/256 rows** (`ann_vamana.cuh:335,354,373`).
   - **Savings:** about -75 s standalone (23%), or about -43 s on top of idea 1.
   - **Effort:** trivial.
   - **Risk:** low-medium. The CheckGraph thresholds (>75% fill, max-degree) are unverified at deg 256 with larger batches. The deg-32 mf=0.1 rows pass.
   - **KEEPS code-path coverage.** The kernels and instantiations are the same, and mf=0.06 stays covered by the deg-32 rows. It does change the parameter point being tested.
   - **Why it works:** batches drop from 22 to 16 and each launch stays latency-bound. In the deg-32 data, mf=0.1 cuts GreedySearch time by 40% (36.1 s to 21.6 s).
3. **Turn the deg-32 2^4 factorial into an 8-run half fraction.** The 4 binary knobs are vs {64,256}, mf {0.06,0.1},
   rb {100,1e6} and iters {1.0,1.5}. Replace the `product` at `ann_vamana.cuh:312-328` with an explicit list that
   keeps even-parity combos (a resolution-IV design: every pair of knobs still appears in all 4 combinations).
   - **Savings:** -59 s (18%).
   - **Effort:** small. **Risk:** low.
   - **REDUCES coverage** of 3- and 4-way interactions only. Every dim and dtype still runs the recall check.
4. **Stop zero-filling a 32 MiB staging buffer per `kvikio_ofstream`** (library). The buffer is
   `buffer_(std::max(cap, ...))`, a `std::vector<char>` at `cpp/src/util/file_io.cpp:265,403`, with a 32 MiB default
   at `include/cuvs/util/file_io.hpp:466`.
   - **Evidence:** there were 2,646 opens (index, dataset, and the sector-aligned files for codebook cases). After each one the main thread is CPU-busy with no traced calls for a median of 9.7 ms (sum 27-30 s), regardless of how few bytes follow.
   - **Change:** use a default-initialised `std::unique_ptr<char[]>(new char[n])` (untouched pages are never faulted), or grow the buffer lazily.
   - **Savings:** about -27 s (8%).
   - **Effort:** small. **Risk:** low.
   - **KEEPS coverage.** It also speeds up every other cuVS serialize test (CAGRA, IVF, brute-force).
5. **Clamp the reverse-batch scratch to N** (library). Set `max_reverse_batch = min(reverse_batchsize, N)` at
   `vamana_build.cuh:253` and use it at `:201` and `:496`. Then lower `PERCENT 100` to about 25 in `CMakeLists.txt:309`.
   - **Savings:** about -0.25 s/process in this executable. Peak memory drops from 8.5 GB to under 0.1 GB, which lets ctest co-schedule other GPU tests for this test's about 317 s. That is a suite wall-time gain, not a per-executable one.
   - **Effort:** small. **Risk:** low (`unique_dests <= N` always).
   - **KEEPS coverage.** It also fixes a real user-facing over-allocation: with the default rb, every build allocates 8 x visited_size MB.
6. **Make the CAGRA recall oracle search all 100 queries in one batch.** Set `max_queries` from 10 to 0 or 100 (`ann_vamana.cuh:325,343,362,381`).
   - **Savings:** about -5 s (1.5%).
   - **Effort:** trivial. **Risk:** low.
   - **KEEPS Vamana coverage.** CAGRA batching is covered by the CAGRA tests.
7. **Remove the per-batch host syncs in the build** (library, `vamana_build.cuh:434-436`, `get_n_components` at `:474`). Use an upper bound (`step_size*degree`) or device-side counts.
   - **Savings:** about -4 to -7 s (about 2%).
   - **Effort:** medium. **Risk:** medium.
   - **KEEPS coverage.**
8. **Raise GreedySearch occupancy for small batches** (library, long term), for example a block or multi-warp per insert when `step_size` is small.
   - **Savings:** potentially most of the 225 s.
   - **Effort:** high. **Risk:** medium-high.
   - **KEEPS coverage.**

Not worth it:
- `naive_knn` and data generation cost less than 0.5 s in total. Caching them per (dtype, dim) would save nothing.
- `CUFileInit` (about 1 s, once per process) is forced by the `CompatMode::OFF` device writes (`src/util/kvikio_io.hpp`), so `KVIKIO_COMPAT_MODE` would not skip it.
- JIT and module loading are 40 ms per process.

Combined estimates:
- Ideas 1+2+3+4: about 323 s down to about 115 s (-64%).
- Coverage-preserving ideas only (2, 4, 6): about 323 s down to about 215 s (-33%), plus the suite-level gain from 5.

Coverage gap worth noting: the deg-256 rows are 43% of the runtime but have no recall check (`ann_vamana.cuh:186`).
They only validate graph fill and serialization.

## Uncertainties

- **The 9.7 ms gap after each `kvikio_ofstream` open is inferred, not measured** (CPU sampling unavailable). What is observed:
  - The gap starts right after `kvikio::FileHandle::FileHandle` returns (`handle_` is constructed before `buffer_` in `sbuf`).
  - No OSRT or CUDA calls occur in the gap, and its length is constant regardless of how much is written next.
  - glibc's internal `mmap` and page faults are not traced, so these observations fit zero-filling a 32 MiB `std::vector<char>` but do not prove it.
- **Savings for idea 2 are extrapolated.** They come from the deg-32 mf ratio and per-launch durations; deg 256 with vs >= N may scale differently.
- **The 8.5 GB peak is computed from the code.** nsys had no `--cuda-memory-usage`. It is consistent with the 100-150 ms pool-growth allocation and the vs=1024-dependent pool-destroy time.
- **Percentages use profiled gtest time.** nsys tracing adds some overhead per kernel launch (about 940 launches/case). The per-process start-up (about 1.05 s) is counted 16 times here but only once under ctest.
- **The suite-level benefit of lowering PERCENT is not measured.** It depends on ctest's resource scheduling and on how much other tests slow down GreedySearch's latency-bound kernels when they share the GPU.
- **The shard-to-case mapping assumes gtest runs cases in index order within a shard.** This was verified by decoding `value_param` in the gtest JSON against `generate_inputs`.
