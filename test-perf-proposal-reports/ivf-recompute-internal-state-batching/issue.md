# IVF: `recompute_internal_state` issues one H2D copy per list pointer

## Summary

`cuvs::neighbors::ivf::detail::recompute_internal_state` (`cpp/src/neighbors/ivf_common.cuh`) updates the
device arrays `data_ptrs` and `inds_ptrs` with one `raft::copy` per list and per array:

```cpp
for (uint32_t label = 0; label < index.n_lists(); label++) {
  auto& list          = index.lists()[label];
  const auto data_ptr = list ? list->data_ptr() : nullptr;
  const auto inds_ptr = list ? list->indices_ptr() : nullptr;
  raft::copy(&data_ptrs(label), &data_ptr, 1, stream);
  raft::copy(&inds_ptrs(label), &inds_ptr, 1, stream);
}
```

That is 2·`n_lists` 8-byte `cudaMemcpyAsync` calls from pageable memory, each costing a few µs of API time. At
`n_lists = 1024`, one call spends ~5 ms on these copies.

## Impact

This function runs after every IVF-Flat, IVF-PQ and IVF-SQ `extend` (and so `build`) and `deserialize`, after
IVF-PQ `extend_list*`/`erase_list` and `clone`, through the public `helpers::recompute_internal_state`, and in
device-side `refine`. Device `refine` builds an IVF-Flat index with `n_lists = n_queries`, so it makes
2·`n_queries` copies per call.

In the gtests under nsys, this costs:

* `NEIGHBORS_ANN_IVF_FLAT_TEST`: ≈ 12 s (1.59 M copies per dtype group);
* `NEIGHBORS_ANN_IVF_PQ_TEST`: ≈ 6 s;
* `NEIGHBORS_ANN_IVF_SQ_TEST`: ≈ 1.8 s.

Applications that call `extend` often, or extend lists one by one, pay the same overhead.

## Proposal

Collect the pointers into two host vectors and upload each array with a single copy on the same stream. The
function already synchronizes the stream before returning, after copying the sorted list sizes to the host. The
host buffers therefore outlive the copies, and the semantics stay the same. The device arrays end up with
identical contents.
