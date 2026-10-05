# Parallelize the IVF-Flat codepacker pack/unpack kernels over elements

`ivf_flat::helpers::codepacker::pack/unpack` launched one thread per row, and each thread looped over all `dim`
components. For typical list sizes (tens to hundreds of rows) that is one mostly idle block per call: 87 µs at dim
2048 and 273 µs at dim 2049 in `NEIGHBORS_ANN_IVF_FLAT_TEST`.

## Changes (`cpp/src/neighbors/ivf_flat/ivf_flat_helpers.cuh`)

* `pack_interleaved_list_kernel` / `unpack_interleaved_list_kernel` now copy one element per thread.
  * `blockIdx.y` selects a group of 32 input rows. It uses a grid-stride loop, with `gridDim.y` capped at 65535.
  * `blockIdx.x` and `threadIdx.x` index the `32 · dim` elements of the group.
  * Threads are enumerated in the destination layout: interleaved order for pack, row-major for unpack. A warp's
    writes are therefore contiguous, and its reads are runs of `veclen` elements.
* The shared helpers `list_position` and `interleaved_offset` replace the per-row device `pack_1`/`unpack_1`. The
  address formula is unchanged. The group offset and `row · dim` are now computed in `size_t`.
* The launch config is `min(256, 32 · dim)` threads per block, with `(ceil(32 · dim / threads), min(ceil(n_rows / 32), 65535))` blocks.
  * It runs on the same stream with no syncs, and the kernel names are unchanged.
  * The launcher now checks the documented precondition `veclen > 0 && dim % veclen == 0`. Before, `veclen == 0`
    hung and a non-dividing veclen accessed out of bounds.

## Why results don't change

This is pure data movement. Each element `(row, l + j)` goes to (or comes from)
`roundDown32(pos) · dim + l · 32 + (pos % 32) · veclen + j`, exactly as before.

* Every destination element is written by exactly one thread, so the order doesn't matter.
* Slots outside the packed rows (before `offset`, the partial last group) are left untouched.

A host emulation of the old and new kernels agrees bit for bit across 61 k configurations. These cover dims 1–4096,
every valid veclen, partial groups, unaligned offsets, explicit indices and the `gridDim.y` stride loop.

## Testing

* `NEIGHBORS_ANN_IVF_FLAT_TEST` (whose packer test round-trips every list through pack and unpack) and the full
  `ctest` suite pass.
* A host emulation of the old and new index mappings (`mapping_check.cpp`) agrees on 61,440 cases: dims 1–4096,
  every valid veclen, partial groups, unaligned offsets and explicit indices.

## Measurements

Single-process wall time of each test executable on an RTX 6000 Ada (48 GB) with a 36-core host, otherwise idle (no ctest parallelism, no MPS). Old and new builds were run alternately, 2 or more repetitions each with the order reversed between repetitions; mean (min–max).

The IVF changes on this branch were measured as a chain: each executable was run against the builds before and after each of them (cumulatively), alternating, 2 repetitions per build. For this PR, "before" is the build just before this change and "after" adds only this change. Only `libcuvs.so` differs for these executables (the test change in between only touches the CAGRA filter-UDF test).

| executable | tests before | tests after | before | after | change |
|---|---|---|---|---|---|
| `NEIGHBORS_ANN_IVF_FLAT_TEST` | 390 | 390 | 112.5 s (112.2–112.7) | 92.0 s (91.1–92.8) | -18.2% |

All tests passed in every run.

Closes #`<issue>`
