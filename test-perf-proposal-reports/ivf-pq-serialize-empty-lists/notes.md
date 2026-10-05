# ivf-pq-serialize-empty-lists (bug found while validating `ivf-list-batched-io`)

* Found by the byte-identity check of `../ivf-list-batched-io/`: the old library crashed (exit 139) in its own
  load-then-save round trip of an IVF-PQ index with empty lists. This happened in the branch baseline (`snap_s0`), in
  `snap_s5` and in `snap_s10`, 3 of 3 runs each. The library with the batched list-I/O change (`snap_s11`, commit
  8904e46f) round-trips the same files byte for byte. That is because its `serialize_lists` treats a null list as
  size 0.
* Load-only works in every build. So the file format is not the problem: the old library reads files written by the
  new one and vice versa (`n_lists=64 size=50`).
* `change.patch` is the minimal fix for `main`, compile-checked against `ivf_pq_serialize.cu` (`-Werror`,
  `-arch=sm_89`; `nvcc -M` confirmed that the edited header was used).
* It is not committed on `test-perf-proposals`, because 8904e46f replaces this loop with `serialize_lists`, which
  already handles null lists. If the batched-I/O PR does not land, this fix should land on its own.
