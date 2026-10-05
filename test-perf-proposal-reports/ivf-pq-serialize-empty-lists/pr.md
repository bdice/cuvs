### Fix `ivf_pq::serialize` for deserialized indexes with empty lists

`ivf_pq::serialize` dereferenced every list pointer, but `deserialize` leaves the pointers of empty lists null, so
saving a loaded index with an empty list crashed. Cast the shared pointer instead and use the `serialize_list`
overload that writes a zero size for a null list, as IVF-Flat and IVF-SQ already do. The file format is unchanged.

## Testing

* Reproducer (build with `add_data_on_build = false`, extend 50 rows into 64 lists, save, load, save): segfaulted
  every time before; the batched list-I/O change on the `test-perf-proposals` branch, which handles null lists the
  same way, round-trips these files byte for byte.
* This patch itself is compile-checked (`ivf_pq_serialize.cu`, `-Werror`); a regression test that saves a loaded
  index with empty lists should be added with it.

Closes #`<issue>`
