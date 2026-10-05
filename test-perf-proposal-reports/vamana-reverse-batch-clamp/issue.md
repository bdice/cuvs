# Vamana build allocates reverse-edge scratch for `reverse_batchsize` rows even when the dataset is smaller

## Describe the bug

`batched_insert_vamana` sizes its reverse-edge buffers from `index_params::reverse_batchsize`. The default is
1,000,000 (`cpp/include/cuvs/neighbors/vamana.hpp:77`), and the buffers are sized this way regardless of the
number of rows in the dataset (`cpp/src/neighbors/detail/vamana/vamana_build.cuh`):

- `rev_ids` and `rev_dists`: `reverse_batchsize x visited_size x 4 B` each (`:303-306`)
- `reverse_list_ptr`: `reverse_batchsize x 32 B` (`:301-302`)
- `s_coords_mem`: `min(10000, max(max_batchsize, reverse_batchsize)) x dim x sizeof(coord)` (`:198-202`)

A reverse batch holds one entry per distinct destination node, so at most N entries are ever used. With the
defaults, every build therefore allocates at least `8 x visited_size` MB of scratch, whatever the dataset size:
- 512 MB at the default `visited_size = 64`
- 8.2 GB at `visited_size = 1024`

## Steps/Code to reproduce bug

Build a Vamana index on a small dataset with a large `visited_size`:

```cpp
cuvs::neighbors::vamana::index_params params;
params.graph_degree = 256;
params.visited_size = 1024;   // reverse_batchsize left at its default of 1e6
auto dataset = raft::make_device_matrix<float, int64_t>(res, 1000, 128);
// ... fill dataset ...
auto index = cuvs::neighbors::vamana::build(res, params, raft::make_const_mdspan(dataset.view()));
```

With the current device resource wrapped in `rmm::mr::statistics_resource_adaptor`, the peak is about 8.2 GB for
this 0.5 MB dataset, and the build fails with out-of-memory on GPUs with less free memory than that. The 1000-row
cases in the test suite show the same thing: `GTEST_CUVS_MEMORY_PEAK=1 ./gtests/NEIGHBORS_ANN_VAMANA_TEST` peaks at
about 8.3 GB (computed from the buffer sizes).

## Expected behavior

Build scratch should scale with the dataset: the reverse batch should never be sized beyond N rows. With
that, the case above needs about 13 MB.

## Additional context

This allocation is also why `NEIGHBORS_ANN_VAMANA_TEST` is registered with `PERCENT 100`
(`cpp/tests/CMakeLists.txt:309`). As a result it holds a whole GPU under ctest for its ~5 min run, although it
launches at most ~15 blocks per kernel.
