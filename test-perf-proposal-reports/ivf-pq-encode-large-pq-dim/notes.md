# ivf-pq-encode-large-pq-dim: IVF-PQ `encode_list_data` kernel is slow at large `pq_dim`

Patch: `change.patch` (one file, `cpp/src/neighbors/ivf_pq/ivf_pq_codepacking.cuh`, 3 loop strides).
`git apply --check` passes. The codes it writes are bit-identical.

## Finding

The hotspot note says `encode_list_data_interleaved_kernel` (`ivf_pq_build.cuh:682`, reached through
`codepacker::extend_list`) takes ~127 ms per launch at `pq_dim` 3072 with ~128 rows, ≈ 26 s in total.
These figures come from the existing nsys summaries (`.../scratchpad/nsys_groups/NEIGHBORS_ANN_IVF_PQ_TEST/*_cuda_gpu_kern_sum.csv`):

| kernel | launches | total | max per launch |
|---|---|---|---|
| `encode_list_data_interleaved_kernel<256, 8>` (i08 big dims, 8-bit) | 7200 | 21.0 s | 146 ms |
| `encode_list_data_interleaved_kernel<256, 6>` (f32 big dims, moderate LUT) | 836 | 4.8 s | 41 ms |
| all `encode_list_data_interleaved_kernel` | | **26.1 s** | |
| `process_and_fill_codes_kernel<256, 8>` (build: **all 4096 rows**, `pq_dim` 3072) | 1279 | 0.29 s | **14.3 ms** |

The build-path encoder uses the same per-(row, subspace) routine (`encode_vectors`) and the same 256-thread blocks with one
warp per row. It encodes 32× more rows than `encode_list_data` in a tenth of the time. Large `pq_dim` doesn't explain the
gap by itself.

## Root cause: block-stride loop over a grid that already covers every row

`write_list`, `write_list_flat` and `run_on_list` (`ivf_pq_codepacking.cuh:220-280`) iterate like this:

```c++
uint32_t stride = subwarp_align::div(blockDim.x);                              // per block, not per grid
uint32_t ix     = subwarp_align::div(threadIdx.x + blockDim.x * blockIdx.x);   // global start
for (; ix < len; ix += stride) { ... }
```

Every launcher sizes the grid to cover all rows: `encode_list_data` launches `ceil(n_rows / 8)` blocks of 8 subwarps, and
pack/unpack/reconstruct launch `ceil(n_rows / 256)` blocks of 256 threads. Block `b` therefore starts at row `8b`
but keeps striding by 8 until `len`, so it handles every row in `[8b, n)`:

* Row `r` is encoded `floor(r/8) + 1` times. That is `n²/16` row-encodes instead of `n`. For `n` = 128 it is
  1088 instead of 128 (8.5× redundant work).
* The critical path is block 0. Each of its subwarps encodes `ceil(n/8)` = 16 rows back to back, and each row is a
  serial loop over all `pq_dim` subspaces.

Per launch: 16 rows × 3072 subspaces × ≈ 2.6 µs per subspace ≈ 128 ms, which matches the measurement. Each subspace
step is latency-bound: 256 codebook entries / 32 lanes = 8 candidate distances with `pq_len` = 2 from an L2-resident
6.3 MB codebook, plus a 5-step shuffle argmin. Only 16 of 142 SMs are busy (8 warps each). Cost grows with list size
× `pq_dim`, so the problem only became visible with the big-dim cases.

Useful work per launch is 128 × 3072 × 256 × 2 ≈ 0.2 G multiply-adds, plus ≈ 0.8 GB of L2 codebook reads. A
well-parallelised kernel could do that in ~0.2–0.5 ms, so the achieved 127 ms is ≈ 300–600× off. The fix recovers the 16×
caused by serialisation and redundancy. The rest comes from low occupancy at small `n_rows` (see "Not done" below).

The same pattern affects pack/unpack/reconstruct (`pack_list_data`, `unpack_list_data`, `{un}pack_contiguous_list_data`,
`reconstruct_list_data`), but only for lists longer than 256 rows. Those are single-block launches in the test
(lists ≈ 128 rows), so they are not hot there. For users they are: unpacking a 100k-row list through
`cuvsIvfPqIndexUnpackContiguousListData` / Python `index.lists()` does ≈ n²/512 ≈ 2·10⁷ row-unpacks instead of 10⁵.

Latent race (fixed as a side effect): `write_vector_flat` (FLAT layout `encode_list_data_flat_kernel`) and
`unpack_contiguous` (via `run_on_list`) write codes with bitfield read-modify-write on shared bytes when
`pq_bits < 8`. Two blocks processing the same row could in principle lose an update. In practice the duplicate passes over a
row are separated in time by one row-encode, so this is unlikely, but it is not guaranteed.

## Fix

Make the three loops grid-stride loops:

```c++
uint32_t stride = subwarp_align::div(blockDim.x) * gridDim.x;   // write_list, write_list_flat
ix += blockDim.x * gridDim.x;                                   // run_on_list
```

Launch configurations are unchanged. With the existing grids each subwarp/thread now runs the loop at most once. A
caller that launches fewer blocks still covers every row exactly once. The stride is at most `n_rows + blockDim - 1`, so
no new overflow is introduced; the start-index arithmetic is untouched.

## Why the codes are bit-identical

* Each (row, subspace) code is still computed by the unchanged `encode_vectors::operator()(i, j)`.
* The lane layout is unchanged: 256-thread blocks, subwarps aligned to `SubWarpSize`, and `lane_id = laneId() % SubWarpSize`.
  So each lane scans the same codebook entries `l = lane, lane + SubWarpSize, ...` in the same order, with the same
  float accumulation over `pq_len`, the same strict `<` first-minimum tie-breaking, and the same `shfl_xor` reduction order.
* Row `ix` still goes to the same `dst_ix` (offset or explicit index), and the chunk/bit positions are unchanged.
* The old code wrote the same bytes 1–16 times (whole 16-byte chunks in the interleaved layout); the new code writes them
  once. The pack/unpack/reconstruct paths are pure data movement and are likewise unchanged per row.

## Expected effect

* Per launch: time drops by ≈ `ceil(n_rows/8)` (≈ 16 for the test's ~128-row lists). At `pq_dim` 3072: 127–146 ms → ≈ 8 ms
  (8-bit) and 35–41 ms → ≈ 2.5 ms (6-bit). After the fix one warp encodes one row, the same as the build kernel, which does
  4096 rows in 14.3 ms.
* Launches that remain per full `NEIGHBORS_ANN_IVF_PQ_TEST` run: only `run()` → `check_lists()` → `check_reconstruct_extend`
  (labels 0, 3, …, 30 = 11 of the 32 lists) calls `extend_list`. Big-dim cases have unique index keys within each TEST_P,
  so the new index cache (60b37464) never hits for them:
  * i08: 4 TEST_Ps (`build_search`, `build_host_input_search`, `build_host_input_overlap_search`, `build_serialize_search`)
    × 10 `big_dims()`.
  * f32: 4 TEST_Ps (`build_host_input_search`, `build_host_input_overlap_search`, `build_extend_search`, `build_serialize_search`)
    × 10 `big_dims_moderate_lut()`.
  * Total: **880 launches with `pq_dim` 256–3072, 88 of them at `pq_dim` 3072**. Filter tests and `build_precomputed` don't
    run the checks.
* GPU time: the big-dim part of the encode time (≈ 16.3 s for i08 + ≈ 4.6 s for f32 ≈ 21 s, unaffected by the cache) drops to
  ≈ 1.3 s. The small-dim part (≈ 5 s before the cache, less now) also drops ~16×, to ≈ 0.1–0.3 s.
  **Expected saving ≈ 20–23 s of GPU time per run.** The host waits for these kernels (`compare_vectors_l2` syncs right
  after), so wall time drops by about the same amount: ≈ 4–5% of the 496 s `ctest -j8` time. The figures come from
  nsys kernel times, which tracing barely inflates.
* Users of `ivf_pq::helpers::codepacker::{extend_list, pack/unpack[_contiguous]_list_data, reconstruct_list_data}` on lists
  longer than one block's worth of rows (8–16 rows for encode, 256 for the others) see the same quadratic → linear change.
  The main `build`/`extend` path (`process_and_fill_codes_kernel`) is not affected.

## Not done (possible follow-up)

Split the subspaces of each row among blocks (`blockIdx.y` over 16-code interleaved chunks, or byte-aligned 8-code groups
for FLAT) when `n_rows` is small. This would bring the post-fix ~8 ms at `pq_dim` 3072 down to ~0.2–0.5 ms, but it saves only
≈ 1–1.5 s more per test run and adds a launch heuristic plus range parameters to `write_vector_*`. It is not worth it in
this patch. The codes would still be identical.

## Affected tests

* `NEIGHBORS_ANN_IVF_PQ_TEST`:
  * `check_reconstruct_extend`: `extend_list` → `encode_list_data`, the hot path.
  * `check_packing`: `pack/unpack_list_data`, with byte-exact `devArrMatch` against the original list.
  * `check_reconstruction`: `reconstruct_list_data`.
  * `flat_layout_codes`: `extend_list_with_contiguous_codes` → `pack_contiguous`.
* Python `test_ivf_pq.py` (`index.lists()` → unpack contiguous) and the C API `cuvsIvfPqIndexUnpackContiguousListData`.
* No test changes are needed. The tests already check that reconstruct → extend reproduces the vectors and that packing
  round-trips byte-exactly.

## Risks

Low. This is a 3-line loop-stride change in device helpers. It changes neither the launch configuration nor the
per-row computation. Every in-tree launcher covers all rows with its grid.

## Build / verification status

* The overlay is a copy of `cpp/src` in `/tmp/ivf-pq-encode-large-pq-dim/ovl`. It was compiled with the exact
  `compile_commands.json` flags (with `-I.../cpp/src` → overlay, `-arch=sm_89`, `-o` in `$TMPDIR`, `nice -n 19`), and
  `nvcc -M` confirmed that the overlay header was picked up and no original `cpp/src` header was. The TUs compiled were
  `ivf_pq_build_common.cu`, `detail/ivf_pq_list_data.cu`, `detail/ivf_pq_contiguous_list_data.cu` and
  `ivf_pq_build_extend_inst_data_f_index_i64.cu`. All compile cleanly with `-Werror`. The only output is nvcc's environmental
  `compiler-bindir` redefinition notice from `NVCC_PREPEND_FLAGS`.
* `clang-format` 20.1.8 with the repo style is clean.
* `git -C /home/coder/cuvs apply --check change.patch` passes.
* Verified on a GPU (commit e239b3fa): all IVF-PQ executables and the full suite pass; `NEIGHBORS_ANN_IVF_PQ_TEST`
  158.6 → 138.5 s (see `pr.md`).

## How to measure

The change is in `libcuvs` only (the TUs above). Rebuild `libcuvs`; the test binary doesn't need recompiling.

* Full: `NEIGHBORS_ANN_IVF_PQ_TEST`.
* Focused: `--gtest_filter='IvfPq/f32_i08_i64.build_search/?'` covers i08 `big_dims()`, indices 0–9, where 6144 is `/9`.
  For f32, `IvfPq/f32_f32_i64.build_extend_search/` indices 9–18 are the moderate-LUT big dims.
* Expected for the i08 `build_search` big-dim cases: ≈ 3.8 s less, from ≈ 4.1 s of encode time to ≈ 0.25 s.
* Under nsys, `encode_list_data_interleaved_kernel<256, 8>` max per launch should drop from ≈ 146 ms to ≈ 8–10 ms.
