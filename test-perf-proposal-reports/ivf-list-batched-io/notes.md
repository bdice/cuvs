# ivf-list-batched-io: notes

Library change to `libcuvs.so` that implements T4(b) of `../PROFILING_SUMMARY.md` (IVF_FLAT idea A, IVF_SQ idea A; IVF-PQ shares the
code). IVF list (de)serialization now uses a few large I/O operations and one stream sync per batch, instead of 2 blocking device
transfers and 2 syncs per list. The on-disk format is byte-identical.

* Files: `cpp/src/neighbors/ivf_list.cuh` (+350 lines, new internal helpers) and the three callers `ivf_flat/ivf_flat_serialize.cuh`,
  `ivf_sq/ivf_sq_serialize.cuh` and `ivf_pq/ivf_pq_serialize.cuh` (each list loop becomes one call). Total +423 / −31 (`change.patch`).
* Not touched: public headers, `ivf_common.cuh`, `util/file_io.cpp` / `kvikio_serialize.hpp`, tests. There is no API or ABI change; the
  new functions live in `cuvs::neighbors::ivf::detail` in a source-tree header.
* `git -C /home/coder/cuvs apply --check change.patch` passes against `test-perf-proposals` (HEAD `7c16a24e`). No other proposal
  directory touches these four files.

## 1. Call path (before the patch, line numbers from HEAD)

### Writing, the same for all three index types

1. The tests call the filename overload:
   * IVF-Flat: `ann_ivf_flat.cuh:237`.
   * IVF-SQ: `ann_ivf_sq.cuh:383`.
   * IVF-PQ: `ann_ivf_pq.cuh:347` (`build_serialize`, used by `TEST_P build_serialize_search`).
   * C API: `c/src/neighbors/ivf_{flat,pq,sq}.cpp:119/110/122`. Python goes through the C API.
2. The filename overload opens a `cuvs::util::kvikio_ofstream`:
   * `ivf_flat_serialize.cuh:85`
   * `ivf_sq_serialize.cuh:75`
   * `ivf_pq_serialize.cuh:105`
3. The stream overload serializes the metadata, centers and `list_sizes`, then loops over the lists:
   * `ivf_flat_serialize.cuh:74-76`
   * `ivf_sq_serialize.cuh:64-66`
   * `ivf_pq_serialize.cuh:74-87`
   Each iteration calls `ivf::serialize_list` (`ivf_list.cuh:153-165` → `:108-151`).
4. The kvikio branch (`ivf_list.cuh:115-129`), for each non-empty list:
   1. `raft::serialize_scalar(size)` is staged in the 32 MiB `kvikio_ofstream` buffer.
   2. `serialize_device_mdspan(data)` (`kvikio_serialize.hpp:58-81`):
      * The numpy header is staged.
      * **`sync_stream`** (`:78`).
      * `write_device` (`:79`) → `sbuf::write_device` (`file_io.cpp:295-312`).
      * **`flush_buffer()`**: a blocking `kvikio::FileHandle::pwrite(...).get()` of the staged header bytes (`:301`, `:399`).
      * **A blocking device `pwrite(...).get()`** on `device_handle_`. Below kvikio's `gds_threshold` this is a POSIX bounce-buffer
        copy + sync + `pwrite`. Above it, a cuFile write runs on the kvikio thread pool while the caller waits on a futex.
   3. The same steps again for the indices.

   In total, every non-empty list costs 2 stream syncs, 2 small blocking host `pwrite`s and 2 blocking device writes. Empty lists only
   stage their scalar.
5. The generic `std::ostream` branch (`ivf_list.cuh:131-150`) is used by the stream overloads and by multi-GPU `iface::serialize`
   (`iface.hpp:224/226`, with a `kvikio_ofstream` from `snmg.cuh:787`, which the `dynamic_cast` catches). Per list, it allocates two
   pageable host arrays, does 2 D2H copies + 1 sync and writes them.

### Reading

1. Entry points:
   * IVF-Flat: `ann_ivf_flat.cuh:239`.
   * IVF-SQ: `ann_ivf_sq.cuh:384`.
   * IVF-PQ: `ann_ivf_pq.cuh:349`.
   * C API: `ivf_{flat,pq,sq}.cpp:127/118/129`.
2. The filename overload creates a `cuvs::util::kvikio_file_reader`:
   * `ivf_flat_serialize.cuh:173`
   * `ivf_sq_serialize.cuh:159`
   * `ivf_pq_serialize.cuh:233`
   `kvikio_file_reader::impl` (`file_io.cpp:210-216`) opens a POSIX fd with an 8 KiB `fd_streambuf` (`file_io.hpp:111`), and eagerly
   opens a compat-off kvikio handle (`open_kvikio_file_for_device_io`, `kvikio_io.hpp:54-72`).
3. `deserialize_impl` reads the metadata, centers and `list_sizes`, then loops over the lists:
   * `ivf_flat_serialize.cuh:154-156`
   * `ivf_sq_serialize.cuh:140-142`
   * `ivf_pq_serialize.cuh:188-203`
4. The reader overload (`ivf_list.cuh:199-222`) does, per list:
   1. Parse the size scalar from the 8 KiB-buffered stream.
   2. `make_shared<ListT>`: a device allocation plus an `kInvalidRecord` fill kernel.
   3. Twice, once for the data and once for the indices: `deserialize_device_mdspan` (`kvikio_serialize.hpp:103-127`), which parses the
      header, does **`sync_stream`** (`:125`) and then `read_device` (`:126`). `read_device` is a `tellg` (lseek), a blocking
      `handle_.pread(dev).get()` (`file_io.cpp:233`) and a `seekg`, which drops the stream buffer, so the next header costs a new `read()`.
5. The generic `std::istream` reader (`ivf_list.cuh:167-197`) is used by user streams and by multi-GPU `std::ifstream` (`snmg.cuh:68`). Per
   list, it allocates host arrays, reads them, does 2 H2D copies and 1 sync.

### cuFile registration

There is **no** per-list buffer registration: kvikio never calls `cuFileBufRegister` on these paths. `cuFileHandleRegister` /
`Deregister` happen once per file handle:
* For writing, when `kvikio_ofstream` lazily opens `device_handle_` on the first `write_device`. That first call is for the centers, not
  for a list.
* For reading, when `kvikio_file_reader` is constructed.

The ≈ 45 ms per case measured in the profiles is therefore a per-file cost. **This patch does not remove it**: the centers still go
through GDS, and the reader still opens its handle eagerly. See "Follow-ups".

## 2. What changed (line numbers are in the patched `ivf_list.cuh`)

| where | what |
|---|---|
| `:243` `kListDirectIoBytes = 4 MiB` | Lists whose payload (data + indices) is larger than this keep the per-list `serialize_list` / `deserialize_list` path, so big lists still go device ↔ file through GDS. |
| `:246` `kListStagingBytes` | The existing 64 MiB `kDeviceSerializationBatchBytes`. Upper bound of the pinned staging buffer. |
| `:260` `list_record_layout`, `:290` `list_record_layouts<ListT>` | Builds the exact bytes of a list record's `head` and `mid` with the writers that `serialize_list` uses, and caches them by list size. `head` is `raft::serialize_scalar(size)` plus `write_numpy_header<value_type>(store extents)`. `mid` is `write_numpy_header<index_type>({size})`. Payload sizes are computed in `size_t`, and an overflow saturates, which forces the per-list path. |
| `:341` `sync_stream_on_exit` | RAII guard: if an exception leaves a function with async copies in flight, it syncs before the pinned buffer is freed. |
| `:367` `serialize_lists` | Works with any `std::ostream`, see below. |
| `:440` `deserialize_lists(std::istream&)` | The old per-list loop; behaviour unchanged. |
| `:469` `deserialize_lists(kvikio_file_reader&)` | The batched reader, see below. |
| the three serializers | Each list loop becomes one call. IVF-PQ passes `static_cast<const list_data_{flat,interleaved}*>(lists()[l].get())` and assigns the deserialized typed list into `impl->lists()`, as before. |

`serialize_lists` works as follows:

1. Pass 1 sums the staged payload bytes and allocates `raft::make_pinned_vector<char>(min(sum, 64 MiB))`, through the handle's pinned
   memory resource.
2. Pass 2 walks the lists in label order:
   * A staged list gets 2 `cudaMemcpyAsync` D2H copies into the buffer.
   * When the buffer would overflow, or before a direct list, the pending lists are flushed: one `sync_stream`, then for each list in
     order `os.write(head)`, data, `os.write(mid)`, indices.
   * A direct list then calls the unchanged `serialize_list`.
   * Empty or null lists only write `head`, the scalar 0.
3. With a `kvikio_ofstream`, all these writes are `memcpy`s into its 32 MiB staging buffer, so the file gets one `pwrite` per 32 MiB.

`deserialize_lists(kvikio_file_reader&)` works as follows:

1. It copies `list_sizes` (already loaded) to the host and gets the file size (`seekg(end)`).
2. Using the expected sizes, it plans batches of consecutive staged records of at most `min(Σ staged records, 64 MiB)` bytes.
3. For each batch:
   * It syncs the previous batch's copies.
   * It does **one** `reader.read_device(pinned, bytes)`. `read_device` accepts host memory, and KvikIO then uses its multi-threaded POSIX
     path.
   * It verifies each record: `head` and `mid` must equal byte-for-byte what the writer emits for the expected size.
   * It allocates the list (`make_shared<ListT>(handle, device_spec, size)`, the same as before) and issues 2 async H2D copies.
4. On the first record that does not match, it `seekg`s to that record's start and hands **all remaining lists** to the old
   `deserialize_list`.
5. Direct lists are read with `deserialize_list`.
6. It syncs once at the end. The stream is left positioned right after the last list, as before.

## 3. Why the format is byte-identical

* **Same bytes.** For every list, the old kvikio branch writes `serialize_scalar(size)`, then
  `write_header(get_numpy_header<T>(shape(store extents), false))` + `write_device(data, prod(extents)·sizeof(T))`, then
  `write_header(get_numpy_header<IdxT>({size}))` + `write_device(indices, size·sizeof(IdxT))`. The old host branch writes the same
  `header_t` via `serialize_host_mdspan` + `os.write`. The new writer emits `head` (the same scalar + the same data header), the same
  device byte range, `mid` (the same indices header) and the same index bytes. The header strings come from the same raft functions with
  the same arguments.
* **Same order.** Records are emitted in label order. Null and empty lists still get only a 0 scalar. `size_override` semantics are
  unchanged: the size written is `sizes_host(label)` for non-null lists.
* **Same stream.** Everything goes to the same `std::ostream` in the same order. `kvikio_ofstream` keeps a single sequential `offset_`, so
  coalescing only changes where `pwrite` calls start and end, not the bytes or their offsets. Lists above 4 MiB call the old function.
* **Host check (`layout_check.cu`).** It was compiled with the libcuvs flags and run on the host only (`CUDA_VISIBLE_DEVICES=`).
  * It checks `head + payload + mid + indices` against the old host branch (`raft::serialize_mdspan`) and against the old kvikio branch's
    header calls, for sizes {0, 1, 7, 31, 32, 33, 100, 1000, 12345}.
  * It covers IVF-Flat f32/dim16, f16/dim3, i8/dim2048 (non-conservative spec), u8/dim4096, IVF-SQ u8/dim13, and IVF-PQ flat 8b×64,
    interleaved 5b×17 and 8b×3072.
  * It also checks that the old parsers (`deserialize_scalar` / `deserialize_mdspan`) read the concatenated records back.
  * Result: all 8 specs are identical.
* **The reader produces the same lists.**
  * It accepts a record only if its header bytes equal what the writer produces for the size in `list_sizes`. That implies every check
    the old parser makes: dtype, rank, shape and size.
  * It copies the same byte ranges to the same destinations: the store-extent prefix of `data`, and exactly `size` indices, so the rest of
    `indices` stays `kInvalidRecord`.
  * Any other file is handed to the old per-list parser from the first deviating record on. That covers a different header formatting, a
    `list_sizes` that disagrees with the records, and truncation. The old parser accepts or rejects it exactly as before. Reads are
    clamped to the file size, so planning never reads past the end.
* **Suggested GPU check (not run here, the GPU was busy).**
  1. Write an index with the old library.
  2. Load it with the new library and save it again.
  3. `cmp` the two files.
  4. Load the new file with the old library and compare search results.

## 4. Memory bounds

* One pinned buffer per call:
  * write: ≤ min(64 MiB, Σ staged payloads);
  * read: ≤ min(64 MiB, Σ staged records).
  Lists above 4 MiB never enter it. The buffer is freed when the call returns.
* The layout cache holds one entry (~200–400 B) per **distinct** list size. d distinct sizes need ≥ d(d−1)/2 rows, so it stays tiny
  (e.g. ≤ 141 k entries for 10¹⁰ rows).
* The reader keeps a host copy of `list_sizes` (4 B × n_lists).
* Before the patch, the kvikio path used no host memory for lists, and the generic path allocated one pageable copy per list.

## 5. Expected effect (estimates from the nsys analyses; not measured, the GPU was reserved)

* **IVF_FLAT.** Serialize + deserialize took 98.3 s nsys (60.1 + 38.2) over 390 cases. Of that, ≈ 45 ms per case (≈ 17.5 s) is the
  per-file cuFile handle cost, which stays. The rest is per list: ≈ 120 µs/list at dim 16, ≈ 880 µs/list at dim 2048, 1024 lists.
  After the patch a list costs ≈ 2 copy enqueues + header `memcpy`s + its share of a few large `pwrite`s / `pread`s, plus the list
  allocation and fill on read, which were there before. **Expect ≈ 55–70 s under nsys (≈ 40–53 s real).** The original 70 s estimate
  assumed the handle cost goes away too.
* **IVF_SQ.** 26 ms fixed + 155 µs per non-empty list per round trip, over 98.6 k lists. **Expect ≈ 12–14 s under nsys (≈ 9–11 s real).**
* **IVF_PQ.** Same path, not costed. Serialize groups take ≈ 3.6–4 s each, including the 1 s `CUFileInit`. Expect a few seconds.
* **Users.** Saving or loading an index with many small or medium lists no longer pays a GDS round trip and 2 syncs per list.
  Indices with lists larger than 4 MiB behave as before for those lists.

## 6. Affected tests (all should pass unchanged; they exercise the new code)

* **C++, file path** (batched writer and batched reader):
  * `NEIGHBORS_ANN_IVF_FLAT_TEST` (`ann_ivf_flat.cuh:237-239`).
  * `NEIGHBORS_ANN_IVF_SQ_TEST` (`ann_ivf_sq.cuh:383-384`).
  * `NEIGHBORS_ANN_IVF_PQ_TEST`: `build_serialize_search`, `ann_ivf_pq.cuh:344-349`, both list layouts if parameterised.
* **Multi-GPU:** `NEIGHBORS_MG_TEST` (`mg.cuh:121-124, 183-186, 296, 346`). The batched writer runs through `std::ostream&` (a
  `kvikio_ofstream`). The reader is the unchanged per-list `std::istream` path (`std::ifstream`).
* **Python:**
  * `test_ivf_flat.py:49-50`, `test_ivf_pq.py:65-66`, `test_ivf_sq.py:52-53, 194-195`.
  * `test_serialization.py` (`test_save_load_ivf_flat`, `test_save_load_ivf_pq`).
  * These go through the C API `cuvsIvf{Flat,Pq,Sq}Serialize/Deserialize`, i.e. the filename path. The C tests
    (`c/tests/neighbors/ann_ivf_*_c.cu`) do not serialize.
* **Coverage gap.** Test lists are ≤ ~0.5 MiB, so the mixed batched + direct (> 4 MiB) path is only hit if k-means leaves a few very
  large lists, which can happen in the InnerProduct dim ≥ 2048 IVF-Flat cases. Ways to cover it:
  * a one-off run with `kListDirectIoBytes = 0` (every list on the old path) and `= SIZE_MAX` (all staged);
  * a dedicated case, e.g. IVF-Flat with n_lists = 4, 20 k rows, dim 256, for ≈ 5 MiB lists.

## 7. Risks

* **GDS use.** Lists ≤ 4 MiB no longer use GDS; they are bounced through pinned host memory. That is also what kvikio does below its
  `gds_threshold` and in compat mode. On a real GDS system, an index made of 1–4 MiB lists may see different throughput; the threshold is
  one constant.
* **Pinned allocation per call.** By default this is a `cudaMallocHost` / `cudaFreeHost`: ≈ ms for tens of MiB. It is negligible against
  the per-list savings, and users can install a pooled pinned resource on the handle.
* **Corrupt `list_sizes`.** With absurd values (≥ 2³² − 31), planning can throw (`round_up_safe` overflow) before any list is read, where
  the old code might only fail later. This only affects corrupt files.
* **Code size.** +350 lines of new code in one header, all in internal templates. The reader's fallback path keeps the old parser as the
  authority. Effort M, risk L–M.

## 8. Smallest worthwhile subset, if reviewers want less

`serialize_lists` alone (no reader change) is about half the code and covers the larger half of the cost: writes are 60 of 98 s in
IVF_FLAT and 15.5 of 24 s in IVF_SQ. The reader can follow separately.

## Follow-ups (not in this patch)

* **Per-file cuFile handle cost (≈ 45 ms/case, T4b/T4c).** Open `kvikio_file_reader`'s GDS handle lazily on the first device read, and
  stage small device arrays (e.g. < 4 MiB centers) through host memory in `serialize_device_mdspan` / `deserialize_device_mdspan`. Then
  IVF files of test size never register a cuFile handle. This changes shared code (CAGRA etc.), so it belongs in its own PR.
* **Generic `std::istream` reader** (multi-GPU `std::ifstream`, user streams). It could read payloads into the pinned buffer and drop the
  per-list syncs as well.
* `cudaMemcpyBatchAsync` (CUDA ≥ 12.8) could replace the 2 copies per list.

## How to measure

Run the executables and compare wall time:
* `NEIGHBORS_ANN_IVF_FLAT_TEST`
* `NEIGHBORS_ANN_IVF_SQ_TEST`
* `NEIGHBORS_ANN_IVF_PQ_TEST`; optionally `--gtest_filter='*build_serialize_search*'`.

Optionally also `NEIGHBORS_MG_TEST` (needs ≥ 2 GPUs) and the Python `test_ivf_*.py` / `test_serialization.py` for correctness. Under
nsys, the per-list `kvikio::FileHandle::pwrite` / `pread` NVTX ranges and `cudaStreamSynchronize` calls in the list loops should drop from
≈ 4 × n_lists to a few per index.

Compile check: 6 TUs built with the exact `compile_commands.json` flags (single `-arch=sm_89`, `-Werror`), no warnings:
* `ivf_flat_serialize_inst_data_{f,h}_index_i64.cu`
* `ivf_sq_serialize_uint8_t.cu`
* `ivf_pq_serialize.cu`
* `ivf_pq_deserialize.cu`
* `ivf_flat_helpers.cu`

The objects contain the `serialize_lists` / `deserialize_lists` instantiations. `layout_check.cu` builds with the
`ivf_flat_serialize_inst_*` compile command (source swapped) and links with plain `nvcc`.
