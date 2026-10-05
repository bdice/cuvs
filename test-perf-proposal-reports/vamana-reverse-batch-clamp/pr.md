# Clamp Vamana's reverse-edge batch to the dataset size

Vamana build sized its reverse-edge scratch (`rev_ids`, `rev_dists`, `reverse_list`, `s_coords_mem`) from
`reverse_batchsize` (default 1e6), even though a reverse batch never holds more than N entries (one per distinct
destination node). This PR clamps the batch size to `min(reverse_batchsize, N)` in one place and uses it for
the allocations and the reverse-batch loop.

**The graph is unchanged.** For `reverse_batchsize >= N`, the reverse loop already ran a single batch of
`unique_dests <= N` entries, so the kernels, grids and counts are the same. For `reverse_batchsize < N`, nothing
changes. No stride depends on the allocated row count. Only unused trailing capacity is removed.

With the scratch bounded, the Vamana test's peak device memory drops from 8.4 GB to 724 MiB (NVML, including the
CUDA context; the RMM peak drops from 7.9 GB to 19 MiB). So its ctest reservation goes from `PERCENT 100` to
`PERCENT 10`, and other tests can share the GPU during its run. 10 is the rule used for the other test
executables: the measured peak × 1.2, as a percentage of a 16 GB GPU, rounded up to a multiple of 5.

## Testing

* `NEIGHBORS_ANN_VAMANA_TEST` passes (all 1,260 cases), as does the full `ctest` suite.
* `compute-sanitizer --tool memcheck` reports no errors on four cases: float and int8, with `reverse_batchsize` 100
  (several batches) and 1e6 (clamped to one batch).

## Measurements

RTX 6000 Ada (48 GB), 36-thread host, otherwise idle. Old and new builds were run alternately, 2 repetitions each.

| `NEIGHBORS_ANN_VAMANA_TEST` | before | after |
|---|---|---|
| wall time, single process | 277.5 s (277.4–277.5) | 277.2 s (277.0–277.3) |
| peak GPU memory of the process (NVML, includes the CUDA context) | 8,596 MiB | 724 MiB |
| peak RMM memory (`GTEST_CUVS_MEMORY_PEAK=1`) | 7,890 MiB | 19 MiB |

The time doesn't change: the oversized buffers were allocated but never touched. The gain is memory. A 1,000-row
build no longer needs more than 8 GB, which also matters to users building small graphs.

**Suite level** (`ctest -j8`, full suite, 2 runs each, alternated): 856.1 / 856.3 s with `PERCENT 100`, 857.9 /
854.2 s with `PERCENT 10`. That is no difference today. Most other executables still reserve the whole GPU, so ctest
can rarely run anything next to Vamana. The lower reservation pays off once the other tests' `PERCENT` values are
right-sized.

Closes #`<issue>`
