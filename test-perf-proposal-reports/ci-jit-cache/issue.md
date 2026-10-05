### C++ test JIT cache is shared by all four test shards, so at most one shard's kernels are cached

**Problem**

#2665 caches `CUDA_CACHE_PATH` between CI runs, with one cache key per CUDA/GPU/driver configuration:
`cache-key-prefix: libcuvs-cudajit-v1` in `.github/workflows/pr.yaml` and `test.yaml`. Since #2720, the C++ tests run
in 4 shard jobs per configuration, and each job runs a different quarter of the test executables, so each links
different JIT-LTO kernels. All 4 jobs compute the same key. Caches are immutable, so the first job to finish saves its
archive and the other three cannot save theirs. On later runs every shard restores that one archive, and three of them
link their kernels again.

**How much it matters**

On an RTX 6000 Ada, the whole suite takes 1,580 s with an empty JIT cache and 856 s with a populated one (`ctest -j8`).
That is ≈ 700 s of linking per configuration, spread over the shards. The CAGRA executables alone account for
+146/+144/+133/+67 s (float, uint8, int8, half), IVF-PQ for +54 s and IVF-Flat for +44 s. A shard that holds a CAGRA
executable but restores another shard's archive pays that cost on every run.

**Proposed fix**

Add the shard to the key prefix
(`libcuvs-cudajit-v1-shard${{ matrix.shard }}of${{ matrix.num_shards }}`), and log the cache file count before and
after the tests so cache hits can be checked in the CI logs.
