### Cheaper inputs for the CAGRA multi-CTA and extreme-inputs bug reproducers

Two reproducers in `NEIGHBORS_ANN_CAGRA_BUGS_TEST` spent nearly all their time on work their bugs do not need. This PR
keeps each bug condition and drops the rest.

**`bug_multi_cta_crash.cu` (#438).** The bug is in the random seed selection. When no sampled node is closer than +inf,
the seed index was left uninitialized and later used to read the graph. The queries are `upper_bound<half>()` (+inf),
so every distance is +inf whatever the graph. The test now builds the index with
`device_padded_index(res, metric, dataset, graph)` from a random `[n_samples, 32]` graph (`raft::random::uniformInt`)
instead of running NN-descent over 1.18M rows. The dataset, queries, search params, graph degree and the separate
search `raft::resources` are unchanged, so the search plan is the same. The test now also checks that every returned
id is a dataset index or the invalid id.

**`bug_extreme_inputs_oob.cu` (#337/#460, #565).** Squared distances of `N(0, 1e20)` data overflow at any row count.
`n_samples` goes from 100k to 10k, and `n_lists` from the default 1024 to 100. That keeps the IVF-PQ statistics per
query the same: ~100 rows per list, ~50 k-means points per cluster, ~2000 candidates for 256 slots. The test used to
accept any `std::exception`. It now requires the `raft::logic_error` raised by the prune check in `graph_core.cuh`
("...invalid or duplicated neighbor nodes..."), and fails if the build returns.

The only coverage lost is the incidental 1.18M-row half NN-descent build. Both tests now run quickly enough to
execute under `compute-sanitizer`.

## Measurements

Single-process wall time of each test executable on an RTX 6000 Ada (48 GB) with a 36-core host, otherwise idle (no ctest parallelism, no MPS). Old and new binaries were run alternately, 2 repetitions each with the order reversed between repetitions; mean (min–max).

Per test (gtest time, both repetitions): `AnnCagraBugMultiCTACrash` 9.99 / 9.96 s → 0.04 / 0.03 s; `cagra_extreme_inputs_oob_test` 2.07 / 1.97 s → 0.74 / 0.76 s. Both changed tests also pass under `compute-sanitizer --tool memcheck` with 0 errors.

| executable | tests before | tests after | before | after | change |
|---|---|---|---|---|---|
| `NEIGHBORS_ANN_CAGRA_BUGS_TEST` | 17 | 17 | 13.8 s (13.6–14.1) | 2.4 s (2.4–2.5) | -82.6% |

All tests passed in every run.

Closes #`<issue>`
