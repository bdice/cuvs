**Title:** Use one parameterized UDF source in the CAGRA filter-UDF test

Each distinct CAGRA filter-UDF source is LTO-linked into the search kernels once per data type. With a cold CUDA JIT
cache that costs ≈ 2 s per (source, data type). This PR replaces the `accept_all`, `reject_all`, `high_filtering_rate`
and `threshold` UDFs with one source:

```cpp
if (filter_data == nullptr) { return true; }
return source_id >= *static_cast<const uint32_t*>(filter_data);
```

Each test passes its bound as a device scalar (`n_rows`, 704 or 192) or `nullptr` (accept all). The tenant UDF stays a
separate source.

- UDF link sets per run go from 6 to 3: float {min_source_id, tenant} and half {min_source_id}. NVRTC compiles go
  from 5 to 2.
- All 24 cases, their assertions and their `filtering_rate` values are unchanged, and the filtered id sets are
  identical. `filtering_rate` does not select different JIT fragments, so no extra links come from it.
- Two different UDFs are still linked for one data type in the same process, so the JIT caches must still key on the
  source.
- Test-only change.

## Testing

All 24 cases pass, with and without a JIT cache.

## Measurements

`NEIGHBORS_ANN_CAGRA_FILTER_UDF_TEST`, the whole executable in one process, on an RTX 6000 Ada with the same
`libcuvs.so` for both builds. "Cold" is a fresh, empty `CUDA_CACHE_PATH` (as in CI); "warm" is a second run with the
same directory. Old and new builds were run alternately, 2 repetitions each.

| | before | after | change |
|---|---|---|---|
| cold JIT cache | 58.9 / 58.2 s | 40.7 / 40.7 s | −30% |
| warm JIT cache | 1.81 / 1.77 s | 1.44 / 1.44 s | −19% |
| JIT cache written by one run | 34.0 MB | 23.3 MB | −32% |

Closes #`<issue>`
