# NEIGHBORS_ANN_HNSW_ACE_FLOAT_UINT32_TEST: where the time goes and how to make it faster

Source: 9 `nsys profile -t cuda,nvtx,osrt` runs, one gtest fixture group per process, on an RTX 6000 Ada with a 36-thread host, branch `staging-test-optimizations-local`. Tests: `cpp/tests/neighbors/ann_hnsw_ace/test_float_uint32_t.cu`. Fixture: `cpp/tests/neighbors/ann_hnsw_ace.cuh` (abbreviated `.cuh` below).
The gtest sum under nsys is 35.9 s; `ctest -j8` took 11.8 s. Every per-group process pays `CUFileInit` (0.97-1.15 s) plus ~3.4 s of start-up/nsys export, but a single ctest process pays `CUFileInit` only once. Removing the 4 extra copies leaves **31.7 s of per-test work**. The four type variants were profiled at overlapping times, so host-latency phases are inflated. ctest/profile ≈ 0.37 for this executable, and the ctest savings below use that factor.

## Summary

| group (TEST_P / TEST) | cases | gtest s | GPU busy | dominant cost |
|---|---|---|---|---|
| AnnHnswAceTest_float.AnnHnswAceBuild | 32 | 26.7 | 4% | 80 ACE partition sub-builds: 18.6 s (NN-descent 9.1 s, reverse-graph loop 6.4 s) |
| AnnHnswAceMemoryFallbackTest | 1 | 3.1 | 5% | 35 partitions forced by a 1 MB host limit: 1.6 s; `CUFileInit` 0.97 s |
| AnnHnswInmemSpillTest | 2 | 2.2 | 1% | `CUFileInit` 0.99 s; 12 spills × ~32 ms buffer zero-fill |
| AnnHnswAceLayeredTest | 1 | 1.9 | 2% | `CUFileInit` 1.02 s; ~0.9 s of real work |
| HnswAceWorkspace.ExistingIndexIsNotTruncated | 1 | 1.5 | 0% | `CUFileInit` 1.04 s |
| CagraAceWorkspace ×2, FileIo, InvalidPartition | 4 | 0.4 | 0% | — |
| **Total** | **41** | **35.9** (31.7 deduplicated) | **~4%** | host-latency bound |

## Where the time goes

The GPU is idle 96% of the time. In AnnHnswAceBuild, kernels total 0.6 s and copies 0.5 s. On the main thread:
* CUDA API calls take 1.8 s.
* `pthread_join` takes 5.8 s. This is waiting for NN-descent's per-iteration `std::thread`.
* Pinned alloc/free takes 0.55 s, for 9 pinned buffers per NN-descent build × 112 builds.
* Most of the rest is untraced host/OpenMP work.

Phases come from the RAFT log timestamps, aligned to the nsys session start, plus kernel markers.

| bucket | s | % of 31.7 | detail |
|---|---|---|---|
| ACE partition sub-builds (`optimize=` in the log) | 18.6 | 59 | 80 builds of 5000 rows (core + augmented), ~230 ms each, with only 0.49 s of kernels |
| ├ NN-descent kNN graph | 9.1 | 29 | ~11 iterations per build. Each iteration spawns and joins a `std::thread` (`cpp/src/neighbors/detail/nn_descent.cuh:2754,2789`). |
| ├ reverse-graph loop in `optimize` | 6.4 | 20 | ACE sub-builds return a host graph, so each of the 64 columns does an OpenMP gather, a 20 KB H2D copy, a kernel and a sync (`cpp/src/neighbors/detail/cagra/graph_core.cuh:836-852`). That is 5120 launches at ~1.2 ms per iteration; fast iterations take <0.1 ms. |
| └ NN-descent finish + sort/prune | 3.0 | 9 | |
| HNSW convert / serialize / deserialize / search | 4.0 | 13 | Upper-layer kNN costs ~2 s: NN-descent on ~150 rows, 32 calls, ~60 ms each. The 32 MiB `kvikio_ofstream` zero-fill takes 0.54 s (16 × 33 ms). Search plus hnswlib I/O is <0.6 s. |
| partition labeling, reorder, read, adjust | 2.9 | 9 | `read` includes the one real `CUFileInit` (1.15 s) |
| test glue (SetUp, naive_knn, eval, temp dirs) | ~0.7 | 2 | not worth optimizing |

* **24 of the 32 AnnHnswAceBuild cases are exact triplicates.** `npartitions` 0, 1 and 2 all resolve to 2 (`ace_resolve_partition_count`, `cpp/src/neighbors/detail/cagra/cagra_build.cuh:1132-1141`). The logged splits are identical: 2376/2624 for L2 and 2333/2667 for IP. The data is seeded, so those cases are 8 configurations run 3 times each.
* **The fallback test hard-codes its limits.** It sets `max_host_memory_gb = 0.001` and `max_gpu_memory_gb = 3.0` (.cuh:423-424), and the input fields at .cuh:1001-1002 are ignored. ACE raises the partition count from 2 to 35 (~286 rows each), and those builds cost 47 ms each.
* **File I/O and start-up are small.**
  * Main-thread file I/O: fwrite 0.35 s (hnswlib `saveIndex`), sendfile 0.17 s (index copy in `hnsw::serialize`), pread/pwrite 0.37 s, statx ×112k 0.15 s, openat ×23k 0.05 s.
  * cuVS has no `fsync` and the OSRT trace shows none. `unlink`/`remove_all` is not traced.
  * JIT/module loading takes 0.06 s.
  * `CUFileInit` includes a 0.3 s `system()` call inside cuFile's handle registration.

## Ideas (ranked by estimated ctest savings out of 11.8 s; they overlap)

1. **Drop the duplicate `npartitions` cases (test-only, keeps coverage).** At .cuh:976, use `{2, 4}`, then append one `npartitions=0` case and one `npartitions=1` case, for example dim 64 L2 in-memory and dim 64 IP disk. That keeps both branches of the count resolution. 32 → 18 cases, removing 10.0 s of profile → **~3.5 s (30%)**. Effort trivial, risk none: the removed builds are byte-identical.
2. **Library: build the reverse graph from a host graph without per-column syncs.** Copy the host graph to the device in a few row or column batches and reuse the device path at `graph_core.cuh:828-834`. This is CAGRA float idea 5, but here 100% of sub-builds take the host path. 6.4 s of profile → **~1-2.4 s (8-20%)**, or ~0.5-1.3 s after idea 1. Keeps coverage. Effort low, risk low.
3. **Library + test environment: skip `CUFileInit` (~1.0 s per process).** The ACE device reads and `kvikio_ofstream::write_device` open files with `CompatMode::OFF` (`cpp/src/util/kvikio_io.hpp:54-72`). That forces cuFile driver initialization even though cuFile then falls back to POSIX here: NVTX shows only `cufile_posix_read`. Make `open_kvikio_file_for_device_io` honour `KVIKIO_COMPAT_MODE=ON`, then set that variable for these tests in `cpp/tests/CMakeLists.txt`. Saves **~1.0 s (8%)**, measured at 0.97-1.15 s in every profile. Reduces coverage of the real GDS path; keep one opt-in GDS job. Effort low.
4. **Library: stop zero-filling the 32 MiB `kvikio_ofstream` staging buffer.** This is CAGRA float idea 6 / Vamana idea 4 (`cpp/src/util/file_io.cpp:265`). Here every `exclusive_hnsw_output_file` (`cpp/src/neighbors/detail/hnsw.hpp:116-121`) pays it: 16 disk-mode serializations plus 12 InmemSpill spills plus 1 fallback, at ~31 ms each. That is 0.92 s of profile, stable across all 4 type profiles → **~0.4-0.9 s (3-8%)**. Keeps coverage. Effort low.
5. **Library: brute-force kNN for small upper HNSW layers.** `all_neighbors_graph` uses NN-descent for anything under 1e7 elements (`hnsw.hpp:486-487`, marked FIXME), even for 150-row levels. A GPU brute-force below ~50k rows costs ~1 ms instead of ~60 ms. ~2 s of profile → **~0.7 s (6%)**. Keeps coverage, and exact upper layers can only help recall. Effort low.
6. **Fallback test: ask for fewer partitions (test-only, keeps coverage).** Set .cuh:423 to ~0.004 GiB. That still has to stay below the in-memory host estimate, which the log prints as 0.01 GiB, so the fallback stays forced; check this first. It gives ~9 partitions instead of 35. Also drop or wire up the dead input fields. 1.2 s of profile → **~0.45 s (4%)**.
7. **Unquantified: OpenMP and thread latency.** The reverse-graph loop does identical work in all four type profiles but took 1.05 s (uint8), 1.5 s (int8), 2.3 s (half) and 6.4 s (float). The partition builds are therefore dominated by fork/join and scheduling latency, not compute. Two OpenMP teams compete in each process: the main thread's, and the NN-descent worker thread's (libomp; 80 parked threads). Try `OMP_NUM_THREADS=4-8` and/or `KMP_BLOCKTIME=0` in the test environment and measure. This needs no code change.

Ideas 1-6 together: **~5-6.5 s (45-55%)**. A further NN-descent fix (a persistent worker instead of a `std::thread` per iteration, or fewer iterations for tiny inputs) would attack the largest remaining bucket (29%). Savings unknown.

## Uncertainties

* CPU sampling was unavailable. Phases come from millisecond log timestamps aligned to `TARGET_INFO_SESSION_START_TIME`, plus kernel and launch markers. The bucket edges are approximate (±10%).
* The ctest factor (0.37) is a whole-executable average. Contention-sensitive phases (ideas 2, 5, 7) were probably inflated more than I/O phases, which is why ranges are given.
* The ~31 ms gap at each `kvikio_ofstream` open is inferred to be the zero-fill. It is a syscall-free window between the sbuf's FileHandle open and the next open, which matches the member order.
* Idea 3 assumes kvikio's compat mode serves device buffers through bounce buffers, which it does, and that a test-only environment variable is acceptable.
* Idea 6: the 11.86 GiB GPU estimate appears to be the RAFT workspace limit (≈25% of 47.4 GiB), not the data size. On small GPUs only the host limit forces disk mode, so keep the host limit below the in-memory estimate.
* Nothing was built or run. Library savings assume the fixed code reaches the fast-iteration cost seen in the same traces.
