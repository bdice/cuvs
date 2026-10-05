# ivf-test-sync-free-checks

Test-only change to `cpp/tests/neighbors/ann_ivf_flat.cuh` and `cpp/tests/neighbors/ann_ivf_pq.cuh`. The test-side
verification of three checks used to make device-to-host round trips per list or per element (each a blocking copy
plus a stream sync). Now it makes a few bulk copies per check. Every check asserts the same conditions with the
same comparators and tolerances on the same data. No test case, parameter, threshold or library file changes.

Background: T5(b) in `../PROFILING_SUMMARY.md`; `NEIGHBORS_ANN_IVF_FLAT_TEST.md` ideas E (test part) and J,
`NEIGHBORS_ANN_IVF_PQ_TEST.md` idea I.

Patch: `change.patch` (2 files, +150/−79 lines). It applies to HEAD `cf8f4927` and to the current working tree
(`git apply --check` passes). Both files are clean under clang-format 20.1.8 with the repo `.clang-format`.

## What changed (line numbers after the change)

### 1. IVF-Flat packer check (`ann_ivf_flat.cuh`, `testPacker`, `:396-492`)

Before, each non-empty list (up to 1024 per case) did the following:

* 5 device allocations;
* gather, `pack`, a `bool` mask by `map_offset`;
* `thrust::reduce` of the mask, which returns to the host (D2H copy and sync);
* 2 `map_offset`s that zero the unmasked elements of the packed list and of the `extend` list;
* `devArrMatch` on them (2 pageable D2H copies of `n_elems` elements, a sync, then a host loop);
* `unpack`;
* `devArrMatch` of the gathered rows against the unpacked rows (2 more D2H copies and a sync).

That is 3 syncs and 5 D2H copies per list.

Now:

* **New helpers** (`:36-79`):
  * `count_if_kernel` / `count_if_async(res, n, pred, count)` adds the number of `i < n` with `pred(i)` to a device
    counter. Each block counts with `__syncthreads_count`, then makes one `atomicAdd`, only if its count is non-zero.
    The helper does not sync.
  * `interleaved_list_mask` is the old mask lambda turned into a functor. It has the same body and the same
    `uint32_t` index type.
* **Allocations outside the loop.**
  * One `n_lists × 3` device counter matrix, zeroed with `cudaMemsetAsync`.
  * Two `max_list_size × dim` buffers for the gathered and the unpacked rows. Each list uses a
    `list_size × dim` view of them.
* **Per list:**
  * the same gather → `pack` → `unpack` calls;
  * three `count_if_async` launches:
    * masked elements (over `n_elems`);
    * masked elements where the packed list differs from the `extend` list (over `n_elems`);
    * elements where the unpacked rows differ from the gathered rows (over `list_size × dim`).
  * No sync, no D2H copy, no allocation.
* **After the loop:** one D2H copy of the counters and one sync. Then a host loop over the labels in the same order
  asserts, for each non-empty list:
  * `n_masked == list_size * dim`;
  * `n_pack_mismatches == 0`;
  * `n_unpack_mismatches == 0`.
* `#include <thrust/reduce.h>` is removed (no longer used). `#include <raft/util/pow2_utils.cuh>` is added (`Pow2` was
  only included transitively).

### 2. IVF-Flat adaptive-centre check (`ann_ivf_flat.cuh`, `buildIndexes`, `:288-334`)

Before, each non-empty list did the following:

* allocate `cluster_data`;
* `copy_selected` and `raft::stats::mean` into a `dim`-sized `centroid`;
* `devArrMatch(centers + dim·l, centroid, dim, CompareApprox<float>(0.001))`: 2 D2H copies and a sync.

Now:

* `cluster_data` is allocated once at the largest list size.
* Each list's mean goes to row `l` of an `n_lists × dim` device matrix `centroids`. The `copy_selected` and `mean`
  calls are the same, with the same arguments.
* After the loop: one D2H copy of `index_2.centers()`, one of `centroids` and one sync.
* A host loop over the non-empty lists then asserts `CompareApprox<float>(0.001)(expected = center, actual = mean)`
  for each component.

### 3. IVF-PQ `compare_vectors_l2` (`ann_ivf_pq.cuh`, `:106-132`)

Before:

* `dist` was a `device_mdarray<double>` backed by a function-local `rmm::mr::managed_memory_resource`, so each call
  did a `cudaMallocManaged` and a `cudaFree`.
* After the kernel, the host loop read `dist(i)`. On a device mdarray that goes through `raft::device_reference`,
  which does an 8-byte `update_host` and a stream sync **per element**: n_rows round trips per call, about 1.9 k per
  index check.

Now:

* `dist` is a plain `make_device_vector<double, uint32_t>`.
* After the same kernel, it is copied once into a `std::vector<double>` and the stream is synced once.
* The loop reads `dist_host[i]` with the same `ASSERT_LE` and the same message.
* `<rmm/mr/managed_memory_resource.hpp>` is dropped and `<vector>` is added.

## Why the checks are equivalent

* **Mask count.**
  * Before: `thrust::reduce(mask, mask + n_elems, 0) == list_size * ps.dim`.
  * Now: `count_if_async(n_elems, mask)` counts the same predicate (same functor body, same `uint32_t` `i`) over the
    same range. The host compares it in 64-bit with `list_size * ps.dim`.
  * Counts are at most `n_elems < 2^32`, so the `uint32_t` counter cannot overflow.
* **Packed data vs the `extend` list.**
  * Before: `Compare<DataT>()(mask[i] ? packed[i] : 0, mask[i] ? extend[i] : 0)` for all `i < n_elems`. Where the mask is
    false, both sides are `DataT{0}`, which always compare equal.
  * Now: the check fails iff some `i < n_elems` has `mask(i) && !(packed[i] == extend[i])`.
  * `Compare<T>` is `a == b` (`test_utils.h:18-20`). For `half` on CUDA < 12.4 it is `float(a) == float(b)`, which is
    the same relation because half → float is exact. The device `==` on `half` is the same IEEE equality.
* **Unpacked vs packed rows.** Before: `devArrMatch(flat_codes, unpacked, list_size * dim, Compare<DataT>())`. Now:
  `count_if(list_size * dim, !(flat_codes[i] == unpacked[i])) == 0`. Same data, same range, same relation.
* **Adaptive centres.**
  * Before: `devArrMatch` with `CompareApprox<float>(0.001)` on `centers[l·dim .. l·dim+dim)` vs `mean(list l)` for
    each non-empty list.
  * Now: the same comparator, called with the same arguments in the same (expected, actual) order, on the same
    rows, for the same lists.
  * The means come from the same kernels with the same arguments. Only the output row changed, from a scratch buffer
    to row `l` of a matrix.
* **`compare_vectors_l2`.** The kernel and the per-row `ASSERT_LE(d, 1.2 * eps * 2^compression_ratio)` with its
  message are unchanged. Only the way `d` reaches the host changed: one bulk copy instead of one copy per element.
  The memory type (managed vs device) does not affect the values the kernel writes.
* **Ordering and buffer reuse.** Everything runs on the fixture's single stream, so reusing the gather, unpack and
  `cluster_data` buffers across lists is stream-ordered. The packed list `L` is not touched after its own checks.
  Each list has its own allocation.
* **Pass/fail is the same.** The test fails iff some non-empty list fails some check. That was true before and is
  still true. The host loops visit the lists and checks in the old order and `ASSERT` on the first failure.

What does differ (diagnostics only):

* **Packer failure messages.** They name the list and the number of mismatching elements (for example, "packed
  data of list 17 differs from the data added with extend", `Which is: 3`), not the first mismatching value and
  offset.
* **Adaptive-centre failure messages.** They keep devArrMatch's `actual=… != expected=… @j` and add `(list l)`.
* **Work after a failure.** After a packer failure in list `L`, the lists after `L` are still packed and checked
  before the assertion fires. Before, the test stopped at `L`.

## Expected effect

Per IVF-Flat packer loop (one per case that is not a cache hit):

* Per list, 3 syncs, 5 D2H copies and 5 allocations are gone.
* With 1024 lists, that is about 3 k syncs, 5 k pageable copies and 5 k allocations per case, replaced by 1 sync and
  1 copy of 12 KiB.
* The D2H volume drops from about 0.7 MB per list at dim 2048 to nothing. The old checks accounted for most of the
  executable's 18.4 GiB of D2H traffic per float group.
* Kernel launches per list go from about 9 to 6.

Per IVF-Flat adaptive-centre check: n_lists syncs and 2·n_lists copies become 1 sync and 2 copies.

Per IVF-PQ `check_lists`:

* About 1.9 k (8-byte D2H + sync) pairs become 1 copy and 1 sync per `compare_vectors_l2` call.
* About 2/3·n_lists `cudaMallocManaged`/`cudaFree` pairs are also gone; `cudaFree` synchronizes the device.

Estimates from the nsys profiles, scaled for the index-reuse changes already on this branch:

* **IVF-Flat packer.** Idea E (test part) was estimated at ≈ 40 s. The packer loop now runs in ≈ 72 of 97 cases per
  data type, which gives ≈ 30 s.
* **IVF-Flat adaptive-centre check.** ≈ 3 s → ≈ 2 s.
* **IVF-PQ.** Idea I was estimated at ≈ 20 s. `check_lists` now runs for 631 instead of 1,397 builds, which gives
  ≈ 9 s.

Wall-clock savings under `ctest -j8` will be smaller, because nsys inflates sync-bound phases. What is left of the
packer loop is mostly the library's slow pack/unpack kernels (≈ 22 s of GPU time, T9) and per-list library work
(`resize_list`, `recompute_internal_state`), which this change does not touch.

## Affected executables (measure these)

* `NEIGHBORS_ANN_IVF_FLAT_TEST`: `ann_ivf_flat/test_{float,half,int8_t,uint8_t}_int64_t.cu`.
* `NEIGHBORS_ANN_IVF_PQ_TEST`: `ann_ivf_pq/test_{float,int8_t,uint8_t}_int64_t.cu`.

Not affected: `NEIGHBORS_ANN_IVF_FLAT_UDF_TEST` and the C API tests `IVF_FLAT_C_TEST` / `IVF_PQ_C_TEST`, which do not
include these fixtures.

## Verification done

* **Compile.** All 7 TUs that instantiate the changed code were compiled: the 4 IVF-Flat data types and the 3 IVF-PQ
  data types.
  * Each used its exact command from `cpp/build/latest/compile_commands.json`, with `-Werror` and
    `-Werror=all-warnings`. All compiled without errors or warnings.
  * Changes to the command:
    * the source and `-I…/cpp/tests` point at an overlay copy of `cpp/tests` under `$TMPDIR`, with `cpp/src`
      symlinked so that `../../src/...` includes resolve;
    * the 7 `--generate-code` flags became a single `-arch=sm_89`;
    * `-o` points into `$TMPDIR`;
    * the command runs under `nice -n 19`.
  * `nvcc -M` confirmed that the overlay's `ann_ivf_flat.cuh`, `ann_ivf_pq.cuh`, `ann_utils.cuh` and
    `test_utils.cuh` were used, and no header from the repo's `cpp/tests`.
* **Patch.** `git -C /home/coder/cuvs apply --check change.patch` passes. Applying the patch to a copy reproduces the
  edited files byte for byte.
* **Not run.** Nothing was run on the GPU. The tests still have to be run to confirm they pass and to measure the
  effect.

## Risks

* **Diagnostics.** Packer failures report counts per list rather than the first mismatching value (see above).
* **New device code in a test header.** `count_if_kernel` is a small templated `RAFT_KERNEL`. Each predicate is its
  own instantiation, and the header is used by 4 TUs of one executable. The kernel uses `__syncthreads_count`, and
  all threads of the block reach it, including those past `n`.
* **Memory.**
  * Packer: two `max_list_size × dim` buffers and `n_lists × 3` counters, instead of five per-list buffers.
  * Adaptive check: `n_lists × dim` floats (8 MiB at 1024 × 2056) on the device and twice that on the host.
  * Both are negligible next to the indexes.
* **Out of scope.** IVF-PQ `check_packing` keeps its 3 `devArrMatch` calls per packed list. Those are already bulk
  copies (2 copies + 1 sync each), they are interleaved with library calls that sync anyway, and they cover only
  1/3 of the lists. Library-side syncs (`recompute_internal_state`, `resize_list`, the codepacker kernels) are T9
  items.
