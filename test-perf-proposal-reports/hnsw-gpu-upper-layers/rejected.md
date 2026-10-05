# Rejected: exact k-NN graphs for small HNSW upper layers

**Proposal.** When the HNSW hierarchy is built on the GPU (`from_cagra` with `HnswHierarchy::GPU`), compute the k-NN
graph of small upper layers exactly instead of running NN-descent or IVF-PQ for each layer. Upper layers keep about
1/M of the rows of the layer below, so most are tiny. The hotspot analysis estimated ≈ 2.5 s of NN-descent setup
across the HNSW tests (`hnsw.hpp:486-487`, a FIXME).

**Result: rejected, negligible effect.** Across the five HNSW executables the change saves 0.5 s of 47.9 s (−1.2%),
within run-to-run noise.

## What was tested

1. **First version: GPU brute force** for layers of up to 16,384 rows. It crashed (SIGSEGV) in
   `NEIGHBORS_HNSW_TEST` and in the HNSW-ACE float and half tests, inside the OpenMP loop of `from_cagra<GPU>` that
   consumes the per-layer graph. The likely cause is that ids returned by the search were cast to `uint32_t` without
   a range check and then used as indices. That version was dropped from the branch.
2. **Second version: exact k-NN on the host** for layers with `n_rows² · dim ≤ 2²⁷`, with ties broken by the
   smaller id. It is correct: the HNSW tests, the four HNSW-ACE executables, all-neighbors, BBQ and `HNSW_C_TEST`
   pass, and a host harness on 768 layer shapes found no out-of-range, self or duplicate ids and matched a
   brute-force reference. It was committed, measured, and then dropped for this rejection.

## Measurements (version 2)

RTX 6000 Ada, 36-thread host, otherwise idle. Single process, `KVIKIO_COMPAT_MODE=ON`. The build before and the
build after (on top of the hnswlib half-distance change) were run alternately, 3 repetitions each; mean (min–max).

| executable | before | after | change |
|---|---|---|---|
| `NEIGHBORS_HNSW_TEST` | 35.3 s (34.6–35.6) | 34.7 s (34.6–34.8) | −1.7% |
| `NEIGHBORS_ANN_HNSW_ACE_FLOAT_UINT32_TEST` | 3.3 s (3.2–3.5) | 3.2 s (3.0–3.4) | −3.4% |
| `NEIGHBORS_ANN_HNSW_ACE_HALF_UINT32_TEST` | 3.1 s (3.1–3.2) | 3.1 s (3.0–3.3) | +0.2% |
| `NEIGHBORS_ANN_HNSW_ACE_INT8_UINT32_TEST` | 3.2 s (3.1–3.3) | 3.4 s (3.1–3.9) | +6.5% |
| `NEIGHBORS_ANN_HNSW_ACE_UINT8_UINT32_TEST` | 3.1 s (3.0–3.2) | 3.0 s (2.9–3.1) | −2.4% |
| **total** | 47.9 s | 47.4 s | −1.2% |

## Why the estimate did not hold

The ≈ 2.5 s estimate came from profiles taken before two changes now on the branch:
* `0bdf0a13` removed the duplicate HNSW-ACE `npartitions` cases.
* `d3bbb9a8` removed the per-iteration NN-descent thread, which cut the cost of each small NN-descent call.

After those, the per-layer NN-descent overhead that this change removes is only a fraction of a second per
executable.

## If revisited

The exact host graph is still a reasonable cleanup of the FIXME (the upper layers become deterministic), but it
should be justified on quality or determinism grounds, not test time. `change.patch` in this directory is version 2.
