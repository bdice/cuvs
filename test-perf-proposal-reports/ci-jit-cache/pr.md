**Title:** Key the C++ test JIT cache per test shard

#2665 keeps the CUDA JIT cache (`CUDA_CACHE_PATH`) of the C++ tests between CI runs. libcuvs links its JIT-LTO
kernels at run time with nvJitLink, and nvJitLink caches the linked kernels there. A CAGRA single-CTA search kernel
takes ≈ 5 s to link, and a cached one takes 20–40 ms.

The C++ tests now run in 4 shards (#2720), and each shard links a different set of kernels. All four use the same cache
key, though. Caches are immutable, so only the first shard to finish saves its archive under that key. The other
three restore kernels they don't use and link their own again on every run.

This PR:

* Adds the shard to `cache-key-prefix` in `pr.yaml` and `test.yaml`
  (`libcuvs-cudajit-v1-shard${{ matrix.shard }}of${{ matrix.num_shards }}`). Each shard then saves and restores its
  own archive.
* Logs the number of cache files before and after the tests in `ci/test_cpp.sh`, so it's visible which shards hit
  the cache.

With the new prefix, the first run on `main` starts each shard from an empty cache and saves four archives.

## Measurements

Whole C++ suite, `ctest -j8`, RTX 6000 Ada, `KVIKIO_COMPAT_MODE=ON`. The empty cache is a fresh `CUDA_CACHE_PATH`, as
in a CI container. The warm cache is a populated cache directory.

| | wall | sum of test times |
|---|---|---|
| warm JIT cache | 857.9 / 854.2 s | 962 s |
| empty JIT cache | 1580.1 s | 1658 s |
| cache written by the cold run | 666,368,737 B in 1,337 files | |

Largest per-test differences, empty minus warm:

| test | extra time with an empty cache |
|---|---|
| CAGRA_FLOAT | +146 s |
| CAGRA_UINT8 | +144 s |
| CAGRA_INT8 | +133 s |
| CAGRA_HALF | +67 s |
| IVF_PQ | +54 s |
| IVF_FLAT | +44 s |
| DISTANCE_TEST | +32 s |
| NEIGHBORS_TEST | +31 s |
| CAGRA_C_TEST | +27 s |
| CAGRA_FILTER_UDF | +26 s |
| DYNAMIC_BATCHING | +17 s |

Cache restored from a tar + `zstd --long=30` archive (what actions/cache does), single process, final build of the
branch:

| executable | empty cache | restored cache | kernels linked after restore | archive |
|---|---|---|---|---|
| `NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST` | 193.0 s | 23.4 s | 0 new cache files | 4.5 MB (118 MB on disk) |
| `NEIGHBORS_ANN_IVF_PQ_TEST` | 194.6 s | 139.2 s | 0 new cache files | 4.7 MB (199 MB on disk) |

Restoring the cache gives warm-cache times: a populated cache brings the whole suite to 857.9 / 854.2 s, against
1580.1 s with an empty one.

**Across builds.** A cache populated by an older build of this branch (`CAGRA_FLOAT`, 196.0 s cold), nine library
commits earlier, was reused by the final build. That run took 40.3 s, and only 74 of 195 cache files had to be
created. The kernels that changed in between were relinked, and the rest were reused. So an archive saved by a nightly
run on `main` saves most of the link time even for PRs that change libcuvs.

**Not measured here.** The CI timing itself: which shards restore an archive that holds their own kernels, and how long a
restore takes. The new log lines report the cache file count before and after the tests in each shard. A shard whose
count grows by hundreds of files linked its kernels again. Measure after the first nightly on `main` has saved the
per-shard archives.

| C++ test shard | PR run before | PR run after |
|---|---|---|
| shard 1 | `<before>` | `<after>` |
| shard 2 | `<before>` | `<after>` |
| shard 3 | `<before>` | `<after>` |
| shard 4 | `<before>` | `<after>` |

Closes #`<issue>`
