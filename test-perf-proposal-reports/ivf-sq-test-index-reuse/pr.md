### Reuse trained IVF-SQ test indices instead of rebuilding them

Each `AnnIVFSQTest` case trained the same quantizer twice: `checkExtend` called `build(add_data_on_build=false)` on the same data and seed.
Each case also rebuilt and re-serialized indices that earlier cases had already built, when the only difference was the search parameters.
k-means and the serialize round trip are ≈ 80 % of the executable's time.

This PR is test-only and changes only `cpp/tests/neighbors/ann_ivf_sq.cuh`:

* **Extend check:** copies the trained `centers` / `sq_vmin` / `sq_delta` into an empty `ivf_sq::index` (the constructor `build` uses) and `extend`s
  it. This replaces the second `build`.
* **Index cache:** the built, the deserialized and the extended index are cached per fixture instantiation.
  * The key is `(num_db_vecs, dim, nlist, metric, host_dataset)`; the database depends only on `(num_db_vecs, dim)`.
  * It is an LRU bounded to 128 MiB of device memory and cleared in `TearDownTestSuite`. Larger entries (dim ≥ 2048) are not cached.
  * Cached indices are only read.
* **Checks unchanged:** every case still computes its naive references and runs the search, serialize, filter and extend checks with its own search parameters,
  with the same thresholds.
* **Duplicate removed:** the exact duplicate input `{1000, 10000, 16, 10, 40, 1024, L2Expanded}` is gone.

| | before | after |
|---|---|---|
| `AnnIVFSQTestF_float` cases | 117 | 116 |
| `AnnIVFSQTestF_half` cases | 17 | 17 |
| total tests | 135 | 134 |
| `ivf_sq::build` calls in parameterized cases | 268 | ≈ 103 |
| serialize round trips | 134 | ≈ 103 |

Float indices `/54`–`/116` shift down by one. No in-repo filters reference them. The parameterized cases no longer call `build` with
`add_data_on_build=false`. That branch only skips the final extend, and `ExtendInPlaceUpdatesListSizeWithinCapacity` still covers it (float, device).

## Measurements

Single-process wall time of each test executable on an RTX 6000 Ada (48 GB) with a 36-core host, otherwise idle (no ctest parallelism, no MPS). Old and new binaries were run alternately, 2 repetitions each with the order reversed between repetitions; mean (min–max).

| executable | tests before | tests after | before | after | change |
|---|---|---|---|---|---|
| `NEIGHBORS_ANN_IVF_SQ_TEST` | 135 | 134 | 35.8 s (35.8–35.8) | 25.3 s (25.0–25.5) | -29.4% |

All tests passed in every run.

Peak GPU memory (RMM peak via `GTEST_CUVS_MEMORY_PEAK=1`; NVML process peak):

| executable | RMM before | RMM after | NVML before | NVML after |
|---|---|---|---|---|
| `NEIGHBORS_ANN_IVF_SQ_TEST` | 528 MiB | 767 MiB | 978 MiB | 1234 MiB |

Closes #`<issue>`
