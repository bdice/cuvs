### [BUG] IVF-PQ codepacker kernels re-process rows in every block, so `extend_list` is quadratic in the list size

**Problem**

`ivf_pq::helpers::codepacker::extend_list` is very slow at large `pq_dim`. In `NEIGHBORS_ANN_IVF_PQ_TEST`
(RTX 6000 Ada, nsys), `encode_list_data_interleaved_kernel` takes **127–146 ms per launch** for a ~128-row list with
`pq_dim = 3072`, and **26 s in total** per run. The build-path encoder `process_and_fill_codes_kernel` uses the same
per-row routine (`encode_vectors`) and the same launch shape, yet it encodes all 4096 rows of that index in 14.3 ms.

**Cause**

`write_list`, `write_list_flat` and `run_on_list` in `cpp/src/neighbors/ivf_pq/ivf_pq_codepacking.cuh` start at the
global row index but step by one *block's* worth of rows:

```c++
uint32_t stride = subwarp_align::div(blockDim.x);
uint32_t ix     = subwarp_align::div(threadIdx.x + blockDim.x * blockIdx.x);
for (; ix < len; ix += stride) { ... }
```

Their launchers already size the grid to cover all rows: `encode_list_data` uses `ceil(n_rows / 8)` blocks, and
pack/unpack/reconstruct use `ceil(n_rows / 256)`. So block `b` processes every row in `[b * rows_per_block, n_rows)`:

* Rows are encoded or (un)packed up to `n_rows / rows_per_block` times, ≈ `n²/(2·rows_per_block)` in total. For a
  128-row list that is 1088 row-encodes instead of 128.
* Block 0 runs the longest: each of its subwarps encodes `ceil(n_rows / 8)` rows back to back, and each row is a serial
  loop over all `pq_dim` subspaces. At `pq_dim` 3072 that is 16 × 3072 latency-bound codebook scans ≈ 127 ms.

The same applies to `pack_list_data`, `unpack_list_data`, `{un}pack_contiguous_list_data` and `reconstruct_list_data`
for lists longer than 256 rows. For example, unpacking a 100k-row list through
`cuvsIvfPqIndexUnpackContiguousListData` does ~2·10⁷ row-unpacks instead of 10⁵.

There is also a latent race. With `pq_bits < 8`, the FLAT-layout encoder and `unpack_contiguous` write codes through
bitfield read-modify-writes on shared bytes, and duplicated processing of the same row by two blocks can lose an update.

**Proposal**

Use grid-stride loops: `stride = subwarp_align::div(blockDim.x) * gridDim.x` in `write_list`/`write_list_flat`, and
`ix += blockDim.x * gridDim.x` in `run_on_list`. Launch configurations and the per-row computation stay as they are,
so the written codes are bit-identical. Every row is just processed once.

Expected effect: per-launch time drops by ≈ `ceil(n_rows / 8)` (≈ 16× for the test's lists, 127 ms → ≈ 8 ms at
`pq_dim` 3072). That removes ≈ 20 s of GPU time from each `NEIGHBORS_ANN_IVF_PQ_TEST` run, and turns quadratic
pack/unpack/encode costs into linear ones for users of the codepacker helpers on large lists.
