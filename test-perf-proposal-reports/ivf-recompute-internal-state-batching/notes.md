# ivf-recompute-internal-state-batching (T9, IVF item)

Patch: `change.patch`, one file: `cpp/src/neighbors/ivf_common.cuh` (+11/-6 lines).
Checks: `git apply --check` passes against `test-perf-proposals` (HEAD `cf8f4927`). The patch does not touch the
uncommitted `cpp/src/cluster/detail/kmeans_balanced.cuh`. Six TUs compile with `-Werror` (listed below). Not run
on a GPU.

## Problem

`ivf::detail::recompute_internal_state` (`ivf_common.cuh:256-292`) refreshes the device arrays of per-list data
and index pointers. It does this with one `raft::copy` per pointer (`:267-273`):

```cpp
for (uint32_t label = 0; label < index.n_lists(); label++) {
  ...
  raft::copy(&data_ptrs(label), &data_ptr, 1, stream);
  raft::copy(&inds_ptrs(label), &inds_ptr, 1, stream);
}
```

That is 2·`n_lists` 8-byte `cudaMemcpyAsync` calls from pageable stack memory per call. At `n_lists = 1024` that is
2048 calls, ≈ 2.5 µs of API time each (≈ 5 ms per call), and more under nsys. The function runs after every
`extend`, `deserialize`, `clone`, `extend_list*` and `erase_list`, and in device-side `refine`.

The analyses measured, under nsys:

* IVF_FLAT: 1.59 M copies and 4.1 s of API time per dtype group; ≈ 12 s in total.
* IVF_PQ: ≈ 220 k copies per group; ≈ 6 s in total.
* IVF_SQ: 584 k copies and 1.56 s of API time in float; ≈ 1.8 s in total.

## Per-list loops found (line numbers are in the unpatched tree)

**Changed (the only per-list single-element H2D loop):** `cpp/src/neighbors/ivf_common.cuh:267-273`. It is a
template on the index type, so this one change covers every IVF flavour that has list pointers:

* IVF-Flat:
  * `ivf_flat_build.cuh:67` (`clone`), `:286` (`extend`, and so `build`) and `:493` (`fill_refinement_index`,
    called from `refine/refine_device.cuh:76` with `n_lists = n_queries`);
  * `ivf_flat_serialize.cuh:159` (`deserialize`);
  * public `ivf_flat::helpers::recompute_internal_state` (`ivf_flat_helpers.cu:189-207`).
* IVF-PQ:
  * `ivf_pq_build.cuh:905/924/943` (`extend_list_with_codes`, `extend_list_with_contiguous_codes`,
    `extend_list`), `:958` (`erase_list`), `:988` (`clone`) and `:1200` (`extend`);
  * `ivf_pq_serialize.cuh:210` (`deserialize`);
  * public `ivf_pq::helpers::recompute_internal_state` (`ivf_pq_build_common.cu:277-280`).
* IVF-SQ: `ivf_sq_build.cuh:359` (`extend`) and `ivf_sq_serialize.cuh:145` (`deserialize`).
* Indirect users: MG (`mg/snmg.cuh`, via IVF-Flat/IVF-PQ build/extend/deserialize), the tiered index
  (`tiered_index.cu`), all-neighbors and CAGRA (`ivf_pq::build`, plus device `refine` in the IVF-PQ graph build,
  `detail/cagra/cagra_build.cuh:2128`).
* IVF-RaBitQ and ScaNN have their own data structures. Neither has a `recompute_internal_state` or a per-list
  pointer upload. Their single-element copies (`ivf_rabitq/gpu_index/ivf_gpu.cu:478,636`,
  `utils/searcher_gpu_utils.cu:59`, `quantizer_gpu.cu:1375`) run once per call, not once per list.

**Reviewed, not changed** (none of them makes one-element H2D copies per list):

* `resize_list` loops: `ivf_flat_build.cuh:280-283` and `:489-491`, `ivf_sq_build.cuh:354-357`,
  `ivf_pq_build.cuh:1185-1196`. They cost one allocation and a fill per list, plus a bulk D2D copy of the old
  contents (`ivf_list.cuh:66-104`). The refine per-query allocation is a separate T9 item.
* Serialize and deserialize loops: `ivf_flat_serialize.cuh:74-76,154-156`, `ivf_pq_serialize.cuh:76-87,191-203`,
  `ivf_sq_serialize.cuh:64-66,140-142`. These do per-list stream I/O with bulk copies (T-level serialization
  item).
* `erase_list` (`ivf_pq_build.cuh:953-956`) and `calculate_offsets_and_indices` (`:244`) each do a single scalar
  copy per call.

## What changed

Inside `recompute_internal_state`:

* The loop now fills two function-scope `std::vector`s of the device arrays' value types,
  `typename decltype(data_ptrs)::value_type` and `typename decltype(inds_ptrs)::value_type`. Each element is
  `list ? list->data_ptr() : nullptr` and `list ? list->indices_ptr() : nullptr`, exactly as before.
* After the loop, two `raft::copy(dst, src, n_lists, stream)` calls replace the 2·`n_lists` single-element ones.
  This is the same pointer overload (`cudaMemcpyAsync(..., cudaMemcpyDefault, stream)`) on the same stream.
* `#include <vector>` is added.

The sort of the cluster sizes, the D2H copy, `sync_stream`, and the `accum_sorted_sizes` loop are untouched.

## Why the results are identical

* Every element of `data_ptrs`/`inds_ptrs` gets the same value as before. The values are computed on the host from
  the same `index.lists()` and don't depend on device state.
* Both copies are enqueued on the same stream, before `sort_cluster_sizes_descending`, as the per-element copies
  were. Stream ordering against earlier and later work is unchanged.
* Lifetime: the host vectors live until the function returns. The function already calls
  `raft::resource::sync_stream(res)` (`:284`) before returning, so the copies have completed before the buffers
  are freed, even if the driver treats a pageable source asynchronously. The original relied on loop-local stack
  variables being staged at call time, which is a weaker guarantee.
* `n_lists == 0` issues a zero-byte copy. That is already the case for the existing D2H `raft::copy` of
  `sorted_sizes`, which uses the same overload.

A pinned buffer was not used. A `cudaHostAlloc` per call would cost more than one staged ~8 KiB pageable copy, and
the stream is synchronized anyway.

## Expected effect

At `n_lists = 1024` each call drops from 2048 copy API calls (≈ 5 ms) to 2 calls plus a 2048-iteration host loop
(≈ 10–20 µs). From the analyses (nsys time; plain wall time should save somewhat less):

* `NEIGHBORS_ANN_IVF_FLAT_TEST` ≈ 12 s (≈ 3 %);
* `NEIGHBORS_ANN_IVF_PQ_TEST` ≈ 6 s;
* `NEIGHBORS_ANN_IVF_SQ_TEST` ≈ 1.8 s (≈ 3 %).

Smaller gains are possible wherever device `refine` runs: `fill_refinement_index` makes 2·`n_queries` copies per
call, for example in the CAGRA IVF-PQ graph build with device data.

## Compile check

Each TU was compiled with its exact `compile_commands.json` command, with these changes:

* `-I.../cpp/src` and `-I.../cpp/include` point to an overlay of `HEAD` (`git archive`) that contains the edited
  header. `nvcc -M` confirmed that `ivf_common.cuh` resolves to the overlay.
* All `--generate-code` flags became a single `-arch=sm_89`.
* `-o` points to `$TMPDIR`, and every command ran under `nice -n 19`.

All compiled cleanly with `-Werror` / `-Werror=all-warnings`. `nm -C` shows the
`recompute_internal_state<...>` instantiation in each object:

* `src/neighbors/ivf_flat/ivf_flat_helpers.cu`: `ivf_flat::index<float|half|int8_t|uint8_t, int64_t>`;
* `build/.../ivf_flat/ivf_flat_build_extend_inst_data_f_index_i64.cu`: `ivf_flat::index<float, int64_t>`;
* `src/neighbors/ivf_pq/ivf_pq_build_common.cu` and
  `build/.../ivf_pq/detail/ivf_pq_build_extend_inst_data_f_index_i64.cu`: `ivf_pq::index<int64_t>`;
* `src/neighbors/ivf_sq/ivf_sq_serialize_uint8_t.cu` and `ivf_sq_build_extend_float_uint8_t_int64_t.cu`:
  `ivf_sq::index<uint8_t>`;
* JIT-LTO fragment `generated_kernels/ivf_sq/scan/ivf_sq_scan_capacity_0_kernel.cu` (includes the header via
  `scan_impl.cuh`), with its own `lto_75` target: OK.

## Tests

Measure (before/after wall time, and nsys `cudaMemcpyAsync` count):

* `NEIGHBORS_ANN_IVF_FLAT_TEST`
* `NEIGHBORS_ANN_IVF_PQ_TEST`
* `NEIGHBORS_ANN_IVF_SQ_TEST`

Correctness (all must pass unchanged):

* IVF suites: the three executables above, plus `NEIGHBORS_ANN_IVF_FLAT_UDF_TEST`. These cover build, extend,
  serialize/deserialize, the packer `helpers::recompute_internal_state` path, and IVF-PQ
  `extend_list`/`erase_list`.
* `NEIGHBORS_TEST`: `refine.cu` runs device refine, which goes through `fill_refinement_index`.
* `NEIGHBORS_TIERED_INDEX_TEST`, `NEIGHBORS_MG_TEST` (IVF-Flat/IVF-PQ MG build/extend/serialize),
  `NEIGHBORS_DYNAMIC_BATCHING_TEST` (ivf_flat/ivf_pq) and `NEIGHBORS_ALL_NEIGHBORS_TEST`.
* One CAGRA executable for the IVF-PQ graph build and device refine, e.g. `NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST`.
* Optional: C tests `ann_ivf_{flat,pq,sq}_c`, and pytest `test_ivf_flat.py`, `test_ivf_pq.py`, `test_ivf_sq.py`,
  `test_refine.py`, `test_mg_ivf_{flat,pq}.py`.

## Risks

Low.

* Host memory: a temporary 16·`n_lists` bytes per call.
* One pageable copy of `n_lists` pointers is staged by the driver just as the single-element copies were.
* No API or ABI change.
* The only behavioural difference is fewer CUDA API calls, which also means fewer `cudaMemcpyAsync` entries in
  nsys/CUPTI traces.
