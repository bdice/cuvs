# IVF-Flat: codepacker pack/unpack kernels use one thread per row

## Summary

`ivf_flat::helpers::codepacker::pack` and `unpack` copy rows between a flat `[n_rows, dim]` matrix and the
interleaved IVF list layout. They do this with `pack_interleaved_list_kernel` / `unpack_interleaved_list_kernel`
(`cpp/src/neighbors/ivf_flat/ivf_flat_helpers.cuh`), which run **one thread per row**. Each thread loops over all
`dim` components:

```cpp
dim3 blocks(raft::div_rounding_up_safe<uint32_t>(n_rows, kBlockSize), 1, 1);  // kBlockSize = 256
...
uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
if (tid < n_rows) { pack_1(codes + tid * dim, list_data, dim, veclen, dst_ix); }  // loops over dim
```

A typical IVF list holds tens to hundreds of rows. For such a list the kernel runs a single block in which only
`n_rows` threads work, each issuing `dim` serial loads and stores. The kernel time therefore grows with `dim`, and
the GPU is almost idle.

## Impact

In `NEIGHBORS_ANN_IVF_FLAT_TEST` (nsys, RTX 6000 Ada), `testPacker` packs and unpacks every list of the index:

* 87 µs per launch at dim 2048 (≈ 10 rows per list, veclen 4);
* 273 µs per launch at dim 2049 (veclen 1);
* **22.4 s of GPU time in total, 30 % of all kernel time** in the executable, on the critical path because the
  test synchronizes after each call.

Users of the public codepacker API (moving vectors into or out of lists) hit the same cost.

## Proposal

Parallelize over elements instead of rows. Assign one thread per element of a group of 32 rows (`blockIdx.y`
selects the row group, grid-stride), and enumerate the threads in the destination layout:

* in the interleaved order for `pack`;
* in row-major order for `unpack`.

Writes are then contiguous per warp and reads are `veclen`-element runs. The element → address formula is the
same as today's, so the result is bit-identical. The public API, the stream semantics (no syncs) and the kernel
names stay the same.

Also worth a look later: `build_index_kernel` (`ivf_flat_build.cuh`) uses the same per-row loop. Its launches are
already large (one per extend batch), and it needs an `atomicAdd` per row, so it is a separate change.
