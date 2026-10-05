### `ivf_pq::serialize` crashes on a deserialized index that has empty lists

**Describe the bug**

Saving an IVF-PQ index that was loaded with `ivf_pq::deserialize` segfaults if any of its lists is empty.

`deserialize_list` leaves the list pointer null for an empty list. The IVF-PQ serializer then dereferences every list
unconditionally (`cpp/src/neighbors/ivf_pq/ivf_pq_serialize.cuh`):

```cpp
auto& typed_list = static_cast<const list_data_flat<IdxT>&>(*index.lists()[label]);   // null for empty lists
ivf::serialize_list(handle_, os, typed_list, list_store_spec, sizes_host(label));
```

`serialize_list` then evaluates `size_override.value_or(ld.size.load())`. `value_or` evaluates its argument eagerly, so
`ld.size` is read through the null reference even though the size (0) was passed explicitly. The same happens in the
interleaved-layout branch. IVF-Flat and IVF-SQ pass the `std::shared_ptr` overload of `serialize_list`, which writes a
zero size for a null list, so they are not affected.

**Steps to reproduce**

1. Build an IVF-PQ index with `add_data_on_build = false` and `n_lists = 64` on 5,000 random rows.
2. Extend it with 50 rows, so most lists stay empty.
3. Serialize it to a file, deserialize it, and serialize it again. The last step segfaults.

The program `ivf_io_check.cpp` in this directory does this
(`ivf_io_check write pq 1 a.bin && ivf_io_check roundtrip pq a.bin b.bin`). It crashed in 3 of 3 runs with the
library built from `main` (October 2026). Deserializing alone works.

**Expected behavior**

The index is saved, and the file matches the one written before loading.

**Fix**

Pass the `std::shared_ptr` to `serialize_list`, like IVF-Flat and IVF-SQ do, so null lists are written as size 0.
