### IVF-Flat C++ tests train the same index three times per case and rebuild it for cases that differ only in search parameters

**Problem**

`NEIGHBORS_ANN_IVF_FLAT_TEST` takes ≈ 275 s under `ctest -j8` (361 s of gtest time under nsys, RTX 6000 Ada). It
has 4 × 97 parameterized cases (`AnnIVFFlatTestF_{float,half,int8,uint8}.AnnIVFFlat`). Every case runs `testIVFFlat()`,
`testPacker()` and `testFilter()`, and each of them builds its own index from the same database:

* **Three k-means trainings per case.** `cpp/tests/neighbors/ann_ivf_flat.cuh:126` (trainset fraction 0.5), `:274`
  (fraction 1.0, non-adaptive) and `:448` (fraction 0.5, `add_data_on_build = true`). k-means is the largest phase
  of the run: **≈ 135 s (37 %)**, 47.2 / 45.3 / 42.5 s for the three call sites. With `n_lists = 1024`, a build runs ≈ 680
  launch-bound EM iterations (≈ 170 µs wall time vs. ≈ 55 µs GPU time each). The second and third training add nothing:
  * `testPacker` only needs the list assignment of an `extend` and a pack target. Neither depends on the trainset
    fraction or on `adaptive_centers`: labels are predicted before adaptive centres are updated.
  * `testFilter`'s `add_data_on_build = true` is just `build` + `extend(dataset, no indices)`
    (`cpp/src/neighbors/ivf_flat/ivf_flat_build.cuh:436-438`). That is the same index `testPacker` builds with
    `extend`.
* **Cases that differ only in search parameters rebuild everything.** `SetUp` draws the database first from a fixed
  seed, so it depends only on `(num_db_vecs, dim)`. The indexes additionally depend on `(nlist, metric,
  adaptive_centers, host_dataset, kernel_copy_overlapping)`. 28 of the 97 cases per data type repeat an earlier
  case's key and differ only in `nprobe`/`num_queries`, e.g. `/28, /30, /32` and `/29, /31, /33`, the host and
  kernel-copy-overlap blocks, and `/95, /96`. Each repeat re-runs all three builds, the kvikio file round trip
  (27 % of the run in total), the per-list packer check (20 %) and four extends. That is ≈ 76 s under nsys.
* **Two exact duplicates.** `inputs` entries `:539` and `:541` are identical to `:535` and `:537`.

Per-phase breakdown (nsys, 4 parameterized groups):

| phase | s | % |
|---|---|---|
| k-means training, 3× per case | ≈ 135 | 37 |
| index file serialize + deserialize | 98.3 | 27 |
| packer check, per-list loop | 72.8 | 20 |
| searches, adaptive-centre check, host recall | 25.8 | 7 |
| extends (4 per case) + packer resize | ≈ 20 | 5.5 |
| naive references, SetUp, teardown | ≈ 9 | 2.5 |

**Proposed fix (test-only, keeps the checks)**

1. Train once per case in `testIVFFlat` and keep the empty trained index. `testPacker` and `testFilter` share
   `extend(database, no indices, trained)`, which is what `add_data_on_build = true` builds. `testPacker` packs into a
   fresh empty index instead of mutating the trained one.
2. Turn the two exact duplicates into dedicated cases (`independent_builds = true`) that keep the old per-sub-test
   builds. This keeps `kmeans_trainset_fraction = 1.0` and the in-place `add_data_on_build` path covered for every
   data type (adaptive and non-adaptive). Test names and the case count stay the same.
3. Keep a byte-bounded (256 MiB) LRU cache per fixture instantiation, keyed on the full build key and cleared in
   `TearDownTestSuite`. It holds the deserialized and the fully extended index of cases that passed all checks. A later
   case with the same key skips the build-only checks (file round trip, centroid invariants, packer), which do not
   depend on the search parameters. It still computes its naive references and runs both searches and evaluations with
   the same thresholds.

**Expected impact**

Per data type, k-means trainings go from 291 to ≈ 76, extends from 388 to ≈ 218, and file round trips and packer
loops from 97 to ≈ 72. Every case still runs its two naive references, searches and recall checks. Estimated
savings: ≈ 130 s of the 361 s under nsys (≈ 36 %), roughly 100 s of the 275 s `ctest` time. The case count (390)
and test names do not change. The cache adds at most 256 MiB of GPU memory. Peak memory for the large high-dim cases
is unchanged.
