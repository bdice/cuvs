### IVF-SQ C++ tests train the same index twice per case and rebuild it for cases that differ only in search parameters

**Problem**

`NEIGHBORS_ANN_IVF_SQ_TEST` takes ≈ 45 s under `ctest` (59.4 s of gtest time under nsys, RTX 6000 Ada). Most of that time goes to building
the same indices over and over:

* **Every case trains two identical quantizers.** `AnnIVFSQTest::testAll` builds with `add_data_on_build = true`.
  `checkExtend` then calls `build_index(false)` (`cpp/tests/neighbors/ann_ivf_sq.cuh:132`), which repeats the same balanced k-means + SQ training
  on the same data with the same `RngState{137}` (`ivf_sq_build.cuh:461`). Only then does it extend. k-means is 40 % of the run (24.0 s), and the second training alone is
  **12.0 s (20 %)**. Each EM iteration is launch-bound (142 µs wall time vs. 47 µs GPU time), and a `nlist = 1024` build runs ≈ 725 of them.
* **Cases that differ only in search parameters rebuild everything.** The database depends only on `(num_db_vecs, dim)`: `SetUp` draws it first
  from a fixed seed. The index additionally depends on `(nlist, metric, host_dataset)`. 134 parameterized cases have only 99 distinct keys.
  Every case still runs the full build, a kvikio serialize/deserialize round trip (≈ 0.2 s at `nlist = 1024`; the round trips total 24 s) and two extends.
  Reusing them across the 35 repeated keys would save ≈ 10 s more.
* `inputs` contains an exact duplicate: `{1000, 10000, 16, 10, 40, 1024, L2Expanded}` at `ann_ivf_sq.cuh:409` and `:449`.

Per-phase breakdown (nsys, both parameterized suites):

| phase | s | % |
|---|---|---|
| k-means training, 2× per case | 24.0 | 40 |
| serialize (kvikio) | 15.5 | 26 |
| deserialize | 8.6 | 14 |
| extends | 4.8 | 8 |
| searches, evals, teardown | 5.1 | 9 |
| naive references, SetUp | 1.3 | 2 |

**Proposed fix (test-only, keeps the checks)**

1. In the extend check, copy the trained quantizer (`centers`, `sq_vmin`, `sq_delta`) of the `add_data_on_build = true` index into an empty
   `ivf_sq::index` (constructed the same way `build` constructs it), then `extend` it. Do not train a second time.
2. Keep a small, byte-bounded LRU cache of the built, the deserialized and the extended index per fixture instantiation.
   Key it on `(num_db_vecs, dim, nlist, metric, host_dataset)` and clear it in `TearDownTestSuite`. Every case still computes its naive references and
   runs all searches and assertions, with the same thresholds, against these indices.
3. Delete the duplicate parameter entry.

**Expected impact**

About 165 fewer k-means trainings, ≈ 31 fewer serialize round trips and ≈ 62 fewer extends per run. That is ≈ 20 s of the 59.4 s under nsys,
roughly 15 s of the 45 s `ctest` time. The case count goes from 135 to 134 (the duplicate). The cache adds at most 128 MiB of GPU memory.
The only API path no longer exercised by the parameterized cases is the `add_data_on_build = false` early-out of `build()` for half/host inputs. The float/device
variant of it remains covered by `ExtendInPlaceUpdatesListSizeWithinCapacity`.
