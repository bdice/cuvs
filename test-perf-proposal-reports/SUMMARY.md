# Test-time proposals: summary

Branch `test-perf-proposals` in `~/cuvs`. One commit per proposal. Each directory here holds `issue.md` and `pr.md` drafts (or
`rejected.md`) plus `notes.md`, and for most library changes a `change.patch` (the patches show each change as first proposed; the
commits are authoritative).

**State on 2026-10-10.** The branch was rebased onto `staging-test-optimizations-local` at 6fbb981c: `upstream/main` f751cadd plus a
local merge of #2699 (still open). Four proposals had merged upstream in the meantime and were dropped from the branch; 21 commits
remain. All of them were remeasured on the new base (next sections). The measurements from before the rebase are kept at the end as
history; the per-proposal `pr.md` tables are from then too.

## Merged upstream (dropped from the branch)

| proposal | upstream | note |
|---|---|---|
| `kvikio-staging-buffer-zero-fill` | #2749 | |
| `vamana-reverse-batch-clamp` | #2747 | Vamana now has `PERCENT 5` in `fe4822c5` (peak 19 MiB) |
| `ivf-pq-encode-large-pq-dim` | #2750 | |
| `nn-descent-host-loop` | #2754 (#2748 closed) | upstream reuses one persistent host worker thread instead |
| (rename only) | #2745 | `NEIGHBORS_ANN_CAGRA_TEST_BUGS` → `NEIGHBORS_ANN_CAGRA_BUGS_TEST`; `gtest_memory_usage.sh` now finds it |

#2726, #2727 and #2729, which the old base merged locally, are upstream too.

## Remaining commits, each measured on its own

Each commit was cherry-picked alone onto the base and built; test binaries and `libcuvs.so` were snapshotted. Every executable a commit
touches was timed against the base, single process, on an RTX 6000 Ada (48 GB, 36 host threads, otherwise idle), with
`KVIKIO_COMPAT_MODE=ON` for both sides. Runs were interleaved, 2 repetitions in reversed order, and the table uses each side's fastest run.
The base's first run in each group was dropped and replaced by 2 more warm runs (its JIT cache was cold: CAGRA float 310 s, then 166 s
on all 3 warm runs). `c4ef930b` only applies on top of `9c688a3c` + `afe7630e`, so it was measured on that pair, 3 repetitions. Seconds
are saved per run of the executables listed; savings overlap, especially on the CAGRA tests. Raw data: `remeasure-2026-10-10/`.

| commit | proposal | kind | lines | saved (s) | where (base → with the commit, s) |
|---|---|---|---|---|---|
| 45220e85 | `kmeans-balanced-iteration-overhead` | lib | +546/−172 | 232 | IVF_PQ 402 → 263, CAGRA float/half/int8/uint8 −28/−23/−13/−14, IVF_FLAT −7, ALL_NEIGHBORS −3 |
| 57b99653 | `cagra-test-index-reuse` | test | +591/−285 | 221 | CAGRA float 166 → 82, half 114 → 47, uint8 111 → 74, int8 106 → 73 |
| 4935488d | `cagra-unaligned-query-batching` | lib | +50/−67 | 160 | CAGRA int8 106 → 61, uint8 111 → 65, float 166 → 123, half 114 → 88 |
| 9c688a3c | `ivf-pq-test-index-reuse` | test | +170/−41 | 131 | IVF_PQ 402 → 271 |
| ebe83ece | `fast-test-verification-helpers` | test | +143/−52 | 87 | IVF_RABITQ 48 → 10, NN_DESCENT 68 → 48, IVF_PQ −20, IVF_FLAT −10 |
| afe7630e | `ivf-flat-test-index-reuse` | test | +331/−141 | 77 | IVF_FLAT 228 → 151 |
| 8ffe8030 | `graph-core-reverse-graph-host-copy` | lib | +75/−19 | 67 | CAGRA float/half/uint8/int8 −25/−19/−13/−10 |
| fab92a69 | `vamana-test-index-reuse` | test | +17/−1 | 37 | VAMANA 284 → 247 |
| 822a88a5 | `ivf-flat-pack-kernels` | lib | +85/−46 | 24 | IVF_FLAT 228 → 204 |
| c4ef930b | `ivf-test-sync-free-checks` | test | +151/−80 | 24 | IVF_FLAT 153 → 130 (after `afe7630e`), IVF_PQ −1 |
| 7b177c9f | `ivf-list-batched-io` | lib | +423/−31 | 20 | IVF_FLAT −13, IVF_SQ 34 → 27 |
| a02dd8e5 | `hnswlib-half-distance` | lib | +239/−2 | 19 | HNSW 70 → 55, HNSW_ACE half 14 → 10 |
| 06376826 | `cagra-filter-udf-single-source` | test | +27/−30 | 19 (cold JIT cache only) | FILTER_UDF 59.2 → 40.4 cold; 3.1 → 2.8 warm |
| 7d4f0909 | `ivf-recompute-internal-state-batching` | lib | +11/−6 | ≤ 16 (noisy) | IVF_FLAT 228 → 219 (other run 241), IVF_PQ −6 |
| 1e65ccaf | `cagra-bug-reproducer-inputs` | test | +50/−16 | 11 | CAGRA_BUGS 13 → 2 |
| 7385f3b8 | `ivf-sq-test-index-reuse` | test | +137/−19 | 10 | IVF_SQ 34 → 24 |
| 4ddf1bc2 | `hnsw-ace-npartitions-dedupe` | test | +10/−2 | 10 | HNSW_ACE ×4 −2…−4 each |
| 273d867e | `cagra-bbq-test-graph-reuse` | test | +139/−22 | 6 | CAGRA_BBQ 10 → 3 |
| 4a701164 | `kvikio-honor-compat-mode` | lib + CI | +25/−2 | 6 | ≈ 1 s per process that does file I/O (UTIL, IVF_SQ, RaBitQ, HNSW_ACE) |
| c8e99eb3 | `ci-jit-cache` | CI | +10/−2 | CI only | see "CI shards and the JIT cache" |
| fe4822c5 | `test-percent-from-peak-memory` | CMake | +36/−36 | suite only | see "Whole suite" |

The new base is faster than the old one: for example IVF_PQ 439 → 402 s, VAMANA 316 → 284 s and CAGRA float 190 → 166 s. These
standalone numbers are not comparable one-to-one with the chain numbers under "History", where each step's saving depended on the
steps before it.

## CI shards and the JIT cache

CI runs the C++ tests in 4 shards (`ctest -j8 -I shard,,4`), each on its own runner. Nightly test run 37894118256 (f751cadd, the
base) shows:

| shard | main tests | RTX PRO 6000 | L4 |
|---|---|---|---|
| 1 | IVF_PQ, CAGRA uint8, HNSW, DISTANCE | 876 s | 1,151 s |
| 2 | NEIGHBORS, RaBitQ, CAGRA bugs/BBQ/merge, MG | 216 s | 225 s |
| 3 | IVF_FLAT, CAGRA float/half, NN_DESCENT, ALL_NEIGHBORS | 835 s | 1,314 s |
| 4 | VAMANA, CAGRA int8, FILTER_UDF, IVF_SQ, DYNAMIC_BATCHING | 655 s | 1,062 s |

All four jobs restore the same 9 MB JIT cache. At the end only shard 2, the shortest and lightest on JIT kernels, saves it. Shards 1, 3
and 4 fail with "Unable to reserve cache with key …, another job may be creating this cache", so the three long shards link their
kernels from scratch on every run. `c8e99eb3` gives each shard its own key.

Locally, each shard's test list from that run was run on the base with the upstream `PERCENT` values, first with an empty JIT cache
and then again with the cache it had just filled (one `CUDA_CACHE_PATH` per shard; `CLUSTER_KMEANS_MNMG_TEST` is not built here):

| shard | empty cache | own cache | saved | cache size (uncompressed) |
|---|---|---|---|---|
| 1 | 899 s | 623 s | −276 s (−31%) | 350 MB |
| 2 | 230 s | 140 s | −90 s (−39%) | 70 MB |
| 3 | 964 s | 667 s | −297 s (−31%) | 219 MB |
| 4 | 748 s | 502 s | −246 s (−33%) | 175 MB |

Shards 1, 3 and 4 run as in the first column today. The restored archive may cover a few of their kernels, so the real gain is
somewhat smaller. Per-shard keys would save about 820 s of GPU-runner time per CI configuration and cut the longest shard by about
300 s, before any test change. The four caches together are about 800 MB per configuration before compression; CI's current
archive is 9 MB compressed.

Mapped onto the shards (local, warm cache, base sums 621 / 138 / 656 / 497 s): shard 3 is the longest, and shard 1 (IVF_PQ alone is
402 s) is next. Savings per shard in seconds:

| commit | shard 1 | shard 2 | shard 3 | shard 4 |
|---|---|---|---|---|
| 45220e85 k-means | 152 | 1 | 64 | 15 |
| 57b99653 CAGRA reuse | 37 | 0 | 150 | 34 |
| 4935488d CAGRA batching | 45 | 0 | 69 | 46 |
| 9c688a3c IVF-PQ reuse | 131 | 0 | 0 | 0 |
| ebe83ece helpers | 20 | 38 | 30 | 0 |
| afe7630e IVF-Flat reuse | 0 | 0 | 77 | 0 |
| 8ffe8030 graph_core | 12 | 1 | 44 | 10 |
| fab92a69 Vamana sweep | 0 | 0 | 0 | 37 |
| 822a88a5 pack kernels | 0 | 0 | 24 | 0 |
| c4ef930b sync-free checks | 1 | 0 | 23 | 0 |

## Whole suite

`ctest -j8` of all C++ tests including the C API tests (59; only the install-header check excluded), warm JIT cache, runs alternated.

**Per-executable `PERCENT` on the base** (`fe4822c5` alone; same binaries, only the generated CTest file swapped):

| `PERCENT` | run 1 | run 2 | mean | sum of per-test times |
|---|---|---|---|---|
| upstream (most at 100) | 2,018 s | 1,928 s | 1,973 s | 2,167 s (run 1) |
| from peak memory | 1,940 s | 1,938 s | 1,939 s (−1.7%) | 11,376 s (run 1) |

On the base the GPU is already the bottleneck. With up to 8 tests sharing it, the long ones run 2–12× slower (VAMANA 296 → 540 s,
IVF_PQ 400 → 1,901 s, CAGRA uint8 111 → 1,330 s), and the suite is as long as IVF_PQ under contention. On the old, optimized
branch the same change gave −14.9%. Its value depends on the others landing first.

**All 21 commits** (branch tip fe4822c5, which includes the `PERCENT` values) against the base with the same `PERCENT` values. Runs
were alternated base, tip, tip, base, after an unrecorded tip warm-up run (775 s, so the JIT cache was warm):

| | run 1 | run 2 | mean | sum of per-test times |
|---|---|---|---|---|
| base + `fe4822c5` | 1,951 s | 1,945 s | 1,948 s | 11,643 s (run 1) |
| branch tip | 775 s | 779 s | **777 s (−60.1%)** | 5,118 s (run 1) |

Against the base with the upstream `PERCENT` values (1,973 s above) that is −60.6%. The longest tests on the tip, under contention,
are IVF_PQ 737 s, IVF_FLAT 652 s, CAGRA float 480 s and Vamana 440 s.

## What to prioritize next

Ranked by saving per line changed, with CI's critical shards (3, then 1) as the tie-breaker.

**Tier 1: small, clear wins. File these first.**

1. `c8e99eb3` `ci-jit-cache` (+10/−2, CI only). The CI logs above show shards 1, 3 and 4 never keep their JIT cache. Locally
   a shard's own cache takes shards 1, 3 and 4 from 899 → 623, 964 → 667 and 748 → 502 s (about −30% each). That is the largest
   saving per line on the branch. Check that per-shard caches (≈ 800 MB per configuration uncompressed) fit the cache backend's
   limits.
2. `4935488d` `cagra-unaligned-query-batching` (+50/−67, removes code): −160 s, all four CAGRA executables −23…−42%, spread over
   shards 1, 3 and 4. It is a library fix: users searching int8/uint8/half or unaligned dims get batched search too.
3. `fab92a69` `vamana-test-index-reuse` (+17/−1): −37 s, Vamana −13%, the longest test in shard 4.
4. `8ffe8030` `graph-core-reverse-graph-host-copy` (+75/−19): −67 s on CAGRA builds; a user-facing build speedup.
5. `4ddf1bc2` `hnsw-ace-npartitions-dedupe` (+10/−2): −10 s across the 4 HNSW_ACE executables; trivial to review.
6. `7d4f0909` `ivf-recompute-internal-state-batching` (+11/−6): small and safe, but the gain (≤ 16 s) is near the noise.

**Tier 2: the largest savings, but larger diffs.** Consider splitting them; see the note below.

7. `9c688a3c` `ivf-pq-test-index-reuse` (+170/−41): −131 s. IVF_PQ is the longest test (402 s, 694 s on L4) and sets shard 1.
8. `45220e85` `kmeans-balanced-iteration-overhead` (+546/−172): −232 s, the largest single saving (IVF_PQ −35%), and it also speeds up
   user index builds. It is a single-file library change, but large; it needs a careful review.
9. `57b99653` `cagra-test-index-reuse` (+591/−285): −221 s, mostly in shard 3, and the biggest diff. It overlaps with 2, 4 and 8,
   which make the searches and builds it removes cheaper.
10. `afe7630e` `ivf-flat-test-index-reuse` (+331/−141): −77 s on IVF_FLAT, the longest test in shard 3. `c4ef930b` (−23 s more)
    depends on it and on 7.
11. `ebe83ece` `fast-test-verification-helpers` (+143/−52): −87 s (RaBitQ −79%, NN-descent −29%), mostly outside the critical
    shards.

**Tier 3: under 25 s each, or poor value per line.** These are `822a88a5` pack kernels (24 s), `c4ef930b` sync-free checks (24 s,
needs 7 and 10) and `7b177c9f` batched IVF I/O (20 s, +423 lines). For the IVF-PQ empty-list crash, file the minimal fix in
`ivf-pq-serialize-empty-lists` instead. Also in this tier: `a02dd8e5` hnswlib half (19 s, +239), `06376826` filter UDF (only with a
cold JIT cache, so `ci-jit-cache` removes most of its value), `1e65ccaf` bug reproducers (11 s), `7385f3b8` IVF-SQ reuse (10 s),
`273d867e` BBQ reuse (6 s) and `4a701164` KvikIO compat mode (6 s, and it changes the CI environment). `fe4822c5` `PERCENT` is
worth −1.7% on today's base, against −15% on the optimized branch; file it after the tier 1 and 2 changes.

**Smaller alternatives to the two test-reuse commits.** Most of `57b99653`'s saving comes from dropping duplicate cases and folding
the U32/I64 twins, not from the index cache; for `9c688a3c` the dedupe is about 40% of the saving in about 45 lines. These are earlier
estimates; neither split was built or measured. A dedupe-only commit for each would be much smaller and could move them up, with the
caches as follow-ups.

## Rejected

| proposal | why |
|---|---|
| `ci-openmp-passive-wait` | `OMP_WAIT_POLICY=passive` makes the suite ~1% slower; NN-descent-style loops +15…+37% (CPU −75%) |
| `hnsw-gpu-upper-layers` | exact k-NN for small HNSW upper layers: −1.2% on the HNSW executables (noise); a GPU brute-force first version crashed |
| `nn-descent-host-loop` part 2 (OpenMP team sizing) | `OMP_NUM_THREADS` experiment: fewer threads never helped wall time; only the thread removal was kept (now upstream as #2754) |

## Findings outside test time

* `ivf-pq-serialize-empty-lists`: `ivf_pq::serialize` segfaults on a deserialized index with empty lists. Still present on `main`
  f751cadd. The minimal fix is in `change.patch`; the batched-I/O commit also fixes it.
* The IVF-PQ codepacking stride bug (`ivf-pq-encode-large-pq-dim`) and the Vamana over-allocation (`vamana-reverse-batch-clamp`) are
  fixed upstream (#2750, #2747).
* CI's JIT cache is saved by one shard of four (above).
* The MG tests peak at 24 GB (`CLUSTER_KMEANS_MG_TEST`) and 38 GB (`NEIGHBORS_MG_TEST`) of RMM memory, more than a 16 GB GPU has.

## Not pursued here (next candidates)

* **Shard balance.** `-I shard,,4` takes every fourth test, so shard 2 gets 216 s and shard 3 gets 835 s on an RTX PRO 6000. Moving one
  long test out of shards 1 and 3, or ordering the tests by cost, would cut the critical shard without touching any test.
* **Contention between concurrent tests** (with `fe4822c5`): single tests slow down 2–12× while sharing the GPU and 36 host threads.
  Splitting the longest executables (IVF-PQ, IVF-Flat, Vamana) would shorten the tail.
* **Vamana** (284 s on the base, 247 s with `fab92a69`): GreedySearch uses one warp per insert and ≤ 15 blocks
  (`vamana_build.cuh:331,341`), and there are per-batch host syncs (`:434-436,474`).
* **IVF-PQ training, option A**: batch all PQ subspaces into one balanced k-means EM loop instead of one fit per subspace.
* **Options that reduce coverage or change the test configuration** (section 3 of `PROFILING_SUMMARY.md`): they need a maintainer
  decision and were left alone.

## History: measurements before the 2026-10-09 rebase

Taken on the older base (0ea21600, then d7fc6586), with the four now-merged commits in the chain. Library changes were measured as a
chain, each build adding one change to the previous one; commit hashes are those of that time.

| # | proposal | kind | effect then (chain, single process) |
|---|---|---|---|
| 1 | `ivf-pq-test-index-reuse` | test | IVF_PQ 437.9 → 280.4 s (−36%) |
| 2 | `cagra-test-index-reuse` | test | CAGRA ×4 569 → 297 s (−48%) |
| 3 | `ivf-flat-test-index-reuse` | test | IVF_FLAT 232.9 → 156.6 s (−33%) |
| 4 | `ivf-sq-test-index-reuse` | test | IVF_SQ 35.8 → 25.3 s (−29%) |
| 5 | `hnsw-ace-npartitions-dedupe` | test | HNSW_ACE ×4 48.4 → 35.8 s (−26%) |
| 6 | `cagra-bbq-test-graph-reuse` | test | BBQ 10.6 → 3.5 s (−67%) |
| 7 | `cagra-bug-reproducer-inputs` | test | TEST_BUGS 13.8 → 2.4 s (−83%) |
| 8 | `fast-test-verification-helpers` | test | RaBitQ −77%, NN_DESCENT −32%, IVF_FLAT/IVF_PQ −7% |
| 9 | `kvikio-staging-buffer-zero-fill` | lib | VAMANA −10%, BRUTE_FORCE −20%, RaBitQ −12%, CAGRA −4…−7% (now #2749) |
| 10 | `graph-core-reverse-graph-host-copy` | lib | CAGRA −5…−8%, HNSW_ACE −1…−12% |
| 11 | `nn-descent-host-loop` | lib | NN_DESCENT −36%, ALL_NEIGHBORS −26%, HNSW_ACE −33…−47%, BBQ −51% (superseded by #2754) |
| 12 | `cagra-unaligned-query-batching` | lib | CAGRA float/half −54%, int8/uint8 −65% |
| 13 | `kmeans-balanced-iteration-overhead` | lib | IVF_PQ −35%, ScaNN −22%, ALL_NEIGHBORS −19%, CAGRA −17…−20% |
| 14 | `ivf-recompute-internal-state-batching` | lib | IVF_FLAT −3%, IVF_SQ −2% |
| 15 | `vamana-reverse-batch-clamp` | lib + CMake | Vamana peak GPU memory 8,596 → 724 MiB; time unchanged (now #2747) |
| 16 | `ivf-test-sync-free-checks` | test | IVF_FLAT −18%, IVF_PQ −5% |
| 17 | `cagra-filter-udf-single-source` | test | FILTER_UDF cold JIT cache 58.6 → 40.7 s (−30%) |
| 18 | `ivf-flat-pack-kernels` | lib | IVF_FLAT −18% |
| 19 | `ivf-list-batched-io` | lib | IVF_SQ −26%, MG −4%; also fixes the IVF-PQ empty-list save crash |
| 20 | `ivf-pq-encode-large-pq-dim` | lib (bug fix) | IVF_PQ −13% (now #2750) |
| 21 | `kvikio-honor-compat-mode` | lib + CI | ≈ 1 s per process that does file I/O (UTIL −65%, HNSW_ACE −14…−26%) |
| 22 | `hnswlib-half-distance` | lib | HNSW −15%, HNSW_ACE half −40% |
| 23 | `vamana-test-index-reuse` | test | VAMANA 276.3 → 241.5 s (−13%), 1,260 → 812 cases |
| 24 | `ci-jit-cache` | CI | CAGRA float 193 → 23 s, IVF_PQ 195 → 139 s with a restored cache; whole suite cold 1,580 s vs warm 856 s |
| 25 | `test-percent-from-peak-memory` | CMake | whole suite 833.9 → 709.4 s (−14.9%) on the optimized branch |

Whole suite then, `ctest -j8` without the C API tests, warm JIT cache: baseline 0ea21600 2,097 s → final branch 828 s (−60.5%).
Per-executable changes from those runs: IVF_PQ 439 → 137 s, VAMANA 316 → 251 s, IVF_FLAT 234 → 78 s, CAGRA ×4 553 → 70 s,
NN_DESCENT 76 → 34 s, HNSW 71 → 34 s, IVF_RABITQ 50 → 10 s, IVF_SQ 36 → 14 s, ALL_NEIGHBORS 30 → 17 s, HNSW_ACE ×4 49 → 12 s.
