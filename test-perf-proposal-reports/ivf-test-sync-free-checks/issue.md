### [TEST] IVF-Flat and IVF-PQ test checks synchronize per list or per element

**Problem**

Three verification checks in the IVF test fixtures make a blocking device-to-host round trip for every list, or for
every element, of the index they check. The GPU work is small, so these tests spend much of their time waiting on
syncs and small pageable copies (nsys on an RTX 6000 Ada):

* **IVF-Flat packer check** (`testPacker` in `cpp/tests/neighbors/ann_ivf_flat.cuh`). For each list (up to 1024 per
  case), the check does the following:
  * a `thrust::reduce` of a mask that returns to the host;
  * a `devArrMatch` of the masked packed data against the masked `extend` data;
  * a `devArrMatch` of the unpacked rows against the packed rows;
  * 5 temporary device allocations.

  That is about 3 syncs and 5 D2H copies per list, and most of the executable's 18.4 GiB of D2H traffic per float
  group. Estimated at **≈ 40 s** of `NEIGHBORS_ANN_IVF_FLAT_TEST`.
* **IVF-Flat adaptive-centre check** (`buildIndexes`). One `devArrMatch` (2 copies + sync) per list:
  **≈ 3 s**.
* **IVF-PQ `compare_vectors_l2`** (`cpp/tests/neighbors/ann_ivf_pq.cuh`). It allocates `dist` from a
  function-local managed memory resource (a `cudaMallocManaged`/`cudaFree` pair per call). It then reads `dist(i)`
  on the host element by element. On a device mdarray, each read is an 8-byte `cudaMemcpyAsync` plus a stream sync:
  about 1.9 k round trips per index check. Estimated at **≈ 20 s** of `NEIGHBORS_ANN_IVF_PQ_TEST`.

The index-reuse changes in these fixtures already run the checks less often. The per-list and per-element round
trips remain on every check that does run.

**Proposal**

This is a test-only change, with the same assertions:

* **Packer check.**
  * Count the masked elements, and the mismatches of the packed and unpacked data, on the device. Store the counts
    in an `n_lists × 3` counter matrix, one small counting kernel per check, with no sync.
  * Copy the counters back once after the loop, and assert per list in the same order. The conditions are the same:
    masked count `== list_size * dim`, zero mismatches.
  * Reuse two gather and unpack buffers sized for the largest list.
* **Adaptive-centre check.**
  * Compute each list's mean with the same `copy_selected` and `mean` calls, into row `l` of an `n_lists × dim`
    matrix.
  * Copy the centres and the means back once, and compare them with the same `CompareApprox<float>(0.001)`.
* **`compare_vectors_l2`.** Use ordinary device memory and copy `dist` to the host once before the per-row
  `ASSERT_LE` loop.

**Expected impact**

* **IVF-Flat packer loop.** About 3 k syncs and 5 k copies per case become 1 of each.
* **IVF-PQ index check.** About 1.9 k round trips become one per `compare_vectors_l2` call.
* **Time saved.** Scaled from the profiles to the index reuse already in place: ≈ 30 s (packer) + ≈ 2 s (adaptive
  centres) in `NEIGHBORS_ANN_IVF_FLAT_TEST`, and ≈ 9 s in `NEIGHBORS_ANN_IVF_PQ_TEST`. Wall-clock savings under ctest
  will be lower, because nsys inflates sync-bound phases.

No test case, parameter, tolerance or library code changes.
