# ivf-flat-pack-kernels: one thread per element in the IVF-Flat pack/unpack kernels

Patch: `change.patch` (only `cpp/src/neighbors/ivf_flat/ivf_flat_helpers.cuh`, +86/−47). Host-side mapping
check: `mapping_check.cpp`.

## Old kernels (HEAD `7c16a24e`)

* `cpp/src/neighbors/ivf_flat/ivf_flat_helpers.cuh`
  * `:19-53`: `__device__ pack_1` / `unpack_1` copy one row. They loop `l = 0..dim` in steps of `veclen` and `j < veclen`,
    and access `block[group_offset * dim + l * 32 + ingroup_id + j]` in 32-bit arithmetic.
  * `:55-84`: `pack_interleaved_list_kernel` / `unpack_interleaved_list_kernel` run **one thread per row**. Each thread
    calls `pack_1`/`unpack_1`, so it makes `dim` serial load/store pairs.
  * `:86-126`: `pack_list_data` / `unpack_list_data` launch `ceil(n_rows / 256)` blocks of 256 threads on the
    resource stream, with no sync. A list of ≤ 256 rows gets one block, and only `n_rows` threads in it do any work.
* The profile (`NEIGHBORS_ANN_IVF_FLAT_TEST.md`) shows 87 µs per launch at dim 2048 (≈ 10 rows, veclen 4) and 273 µs at
  dim 2049 (veclen 1). In total that is **22.4 s of GPU time, 30 % of all kernel time** in the executable (nsys, 4 dtype groups).

## Callers

* Public API: `ivf_flat::helpers::codepacker::pack` (`cpp/include/cuvs/neighbors/ivf_flat.hpp:2512,2542,2572,2602`)
  and `unpack` (`:2635,2668,2700,2733`), for float/half/int8/uint8 with `IdxT = int64_t`.
  → `cpp/src/neighbors/ivf_flat/ivf_flat_helpers.cu:17-103`
  → `detail::pack/unpack` (`ivf_flat_helpers.cuh:131-153`, always with a `uint32_t offset`)
  → `pack_list_data/unpack_list_data`
  → the kernels. `ivf_flat_helpers.cu` is the only file that includes `ivf_flat_helpers.cuh`.
* In the repository, only the test calls these functions: `AnnIVFFlatTest::testPacker`
  (`cpp/tests/neighbors/ann_ivf_flat.cuh:355` pack and `:414` unpack). It calls them once per non-empty list, with
  `offset = 0` and `n_rows = list size`. That is ≈ 10 rows per list for 10 k rows / 1024 lists, and ≈ 100–130 rows
  for 100 k–131 k rows. Dims range from 1 to 4096 (2048–2056 in the slow cases), and veclen is 1, 4, 8 or 16
  (`index::calculate_veclen`). After commit `9e297954`, cases served from the index cache skip `testPacker`. The
  high-dim indexes (≈ 256 MB) don't fit the 256 MiB cache, so the slow cases still run it.
* Not callers: serialization (`ivf_flat_serialize.cuh`, `ivf_list.cuh`) copies the interleaved list buffers verbatim.
  `extend`/`build` write lists with `build_index_kernel` (see below). The C, Python, Java, Rust and Go bindings don't
  expose the IVF-Flat codepacker. The host `pack_1`/`unpack_1` helpers (`ivf_flat_helpers.cuh:155-188`) are separate
  and unchanged.

## New mapping

Both kernels use one thread per element. `blockIdx.y` (grid-stride, capped at 65535) selects a group of 32
consecutive **input** rows. `ix = blockIdx.x * blockDim.x + threadIdx.x` indexes the `32 * dim` elements of that
group. Threads are enumerated in the **destination** layout, so consecutive threads write consecutive elements:

* pack (writes the list): `ix → (chunk = ix / (32·veclen), row_in_group, j)` with
  `l = chunk·veclen`. The thread copies `codes[row·dim + l + j]` to
  `list[roundDown32(pos)·dim + l·32 + (pos % 32)·veclen + j]`.
* unpack (writes `codes`): `ix → (row_in_group = ix / dim, c = ix % dim)`, `j = c % veclen`, `l = c − j`, with the same
  two addresses swapped.
* `row = group·32 + row_in_group` (skip if `row ≥ n_rows`), and `pos = offset + row` (or `indices[row]`).
* Launch: `threads = min(256, 32·dim)` (a multiple of 32), `blocks = (ceil(32·dim / threads), min(ceil(n_rows / 32), 65535))`.
  It stays on the same stream, with no sync. Kernel names are unchanged, so nsys reports stay comparable.
* With `offset % 32 == 0` (the test and the usual append case), the writes of a warp are one contiguous run in
  both directions. The reads are runs of `veclen` elements (16 bytes when veclen > 1). With an unaligned offset, a
  pack warp writes at most two runs.
* Address math uses `size_t` for `row·dim` and `roundDown32(pos)·dim`. The old code wrapped at 2³² elements.
* Resource usage on sm_89: 15–16 registers, no stack, no spills.

## Why the results are identical

It is pure data movement with the same address formula. For an element `(row, l + j)` the old code wrote
`group_offset·dim + l·32 + ingroup_id + j`, with `group_offset = roundDown32(pos)` and `ingroup_id = (pos % 32)·veclen`.
That is exactly `interleaved_offset(pos, dim, veclen, l, j)`. The new enumeration is a bijection between thread
indices `{(group, ix) : row < n_rows}` and the old `{(row, l, j)}` triples. Every destination element is written by
exactly one thread, and positions are distinct (`offset + row`, or distinct `indices`), so thread order can't matter.
Slots of the list that don't belong to the packed rows (before `offset`, after the last row, other rows of a
partial group) are never touched, as before. The only difference is that indices ≥ 2³² no longer wrap: the new
code is correct where the old one overflowed.

`mapping_check.cpp` checks this. It transcribes the old and new kernel bodies verbatim and runs them over the
launch grid on the host. Sources hold unique values and destinations hold a sentinel. The check then asserts
bit-identical destinations, no out-of-bounds reads or writes, and no double writes. It covers:

* dims {1–6, 8, 12, 16, 17, 24, 32, 33, 48, 64, 100, 128, 136, 256, 1000, 1024, 2048, 2049, 2056, 4096};
* every veclen in {1, 2, 4, 8, 16} that divides dim;
* n_rows {1, 2, 7, 31, 32, 33, 63, 64, 65, 100, 257, 1000, 3000} (≤ 2¹⁹ elements);
* offsets {0, 1, 5, 31, 32, 33, 42, 64, 100}, plus a random injective `indices` map;
* `gridDim.y` caps {65535, 1, 3} (exercises the grid-stride loop);
* both pack and unpack.

Result: **61 440 cases, 2.17 G element copies, 0 failures** (`g++ -O2 -std=c++17`, ~40 s). Mutation sanity: an
off-by-one row bound, a wrong grid stride and a swapped in-chunk index each produce thousands of failures.

## Expected effect

* A call now launches `32·dim` threads per 32-row group instead of `n_rows` threads looping over `dim`. At dim 2048
  and 10 rows that is 65 k threads moving 80 KB, which should take a few µs (launch-bound) instead of 87 µs, or
  273 µs at dim 2049.
* The ≈ 22 s of pack/unpack GPU time in `NEIGHBORS_ANN_IVF_FLAT_TEST` (under nsys, before the test-side index
  reuse) should drop to ~1–2 s. The test syncs after every pack (`thrust::reduce`) and unpack (`devArrMatch`), so
  kernel time is on the critical path and the wall-clock saving should be close to the GPU-time saving. Most of
  it comes from the dim 2048–2056/4096 cases.
* Users of `codepacker::pack/unpack` (e.g. moving data into or out of lists) get the same speedup.

## Affected tests / executables to measure

* `NEIGHBORS_ANN_IVF_FLAT_TEST`: correctness through `testPacker` (pack is compared against `extend`'s layout and
  unpack against the source rows), plus timing. The nsys kernel names are unchanged:
  `pack_interleaved_list_kernel`, `unpack_interleaved_list_kernel`.
* No other test executable reaches this code.

## build_index_kernel (not changed)

`cpp/src/neighbors/ivf_flat/ivf_flat_build.cuh:105-155` has the same one-thread-per-row interleaving loop
(`:150-154`). It is launched with `ceil(rows / 256)` blocks in `extend` (`:311-323`, per batch) and in
`fill_refinement_index` (`:497-507`, `n_queries · n_candidates` rows). I left it alone:

* each row claims its in-list slot with an `atomicAdd`, so per-element threads would need a warp-per-row scheme
  (lane 0 does the atomic, then a shuffle broadcast) or a two-pass scheme;
* its launches already cover thousands to millions of rows, so the GPU is full;
* it doesn't appear as a hotspot in the profile.

It is a possible follow-up for coalescing on large extends.

## Risks

* Low. It is pure data movement on the same stream, with the same kernel names and public API, and no new
  allocations or syncs.
* New host-side `RAFT_EXPECTS(veclen > 0 && dim % veclen == 0)` in the launcher. That is the documented
  precondition. Previously, `veclen == 0` hung the kernel and a non-dividing veclen read and wrote out of bounds.
  `index.veclen()` always satisfies it.
* For lists with fewer than 32 rows, most lanes of the 32-row group are idle (up to 31/32). That is still only
  `32·dim` threads per call.
* `32·dim` must fit in `uint32_t` (dim < 2²⁷). The old code already overflowed `group_offset·dim` in that range.
* The old kernel read `indices[tid]` before its `tid < n_rows` check (an out-of-bounds read in tail threads). The
  indices variant is unused by any caller, and the new code only reads it for valid rows.
* Needs a GPU run of `NEIGHBORS_ANN_IVF_FLAT_TEST` to confirm. It was not run here: compile-checked only.

## Verification done here

* `ivf_flat_helpers.cu` (the only TU that includes the header) compiles with the exact build command from
  `compile_commands.json` with `-arch=sm_89` (`-Werror`, `-Wall`, `--expt-relaxed-constexpr`). `nvcc -M` confirmed
  that the edited overlay header was used. There were no warnings apart from the environment's
  `NVCC_PREPEND_FLAGS -ccbin` redefinition notice, which every build here prints.
* `clang-format` 20.1.8 with `cpp/.clang-format`: clean.
* `git -C /home/coder/cuvs apply --check change.patch`: OK. Applying the patch to a pristine copy reproduces the
  compiled file byte for byte.
