# Remeasurement on the rebased branch (2026-10-10)

Raw data behind the tables in `../SUMMARY.md`. Base: `staging-test-optimizations-local` 6fbb981c (`upstream/main` f751cadd + #2699).
RTX 6000 Ada, single process unless noted.

* `standalone.tsv`: each remaining commit cherry-picked alone onto the base, timed against the base on every executable it touches
  (2 interleaved repetitions, order reversed in the second; `KVIKIO_COMPAT_MODE=ON`). Columns: rep, exe, variant, wall_s, rc, load1,
  tests, failures, gtest_s. The base's first run in each group had a cold JIT cache; `summarize_standalone.py` drops it.
* `base_recheck.tsv`: 2 more warm-cache runs of the base for every executable above.
* `syncfree_chain.tsv`: `c4ef930b` on top of `9c688a3c` + `afe7630e` (`sf_chain`) against that pair (`sf_pre`), 3 repetitions.
* `suite_runs.txt`: FILTER_UDF cold/warm JIT cache; whole suite with upstream vs per-executable `PERCENT` on the base; base +
  `PERCENT` vs the branch tip; CI's four shards with an empty and then their own JIT cache.
* `ci_shards_f751cadd.txt`: the test list of each CI shard in nightly run 37894118256.
* `summarize_standalone.py`, `shard_impact.py`: the tables in the summary. They read the original layout (`ab_standalone/results.tsv`,
  `ab_recheck/`, `ab_syncfree/`, `ci_logs/shard*.txt`, `suite_std_base_r2.log`); adjust the paths to rerun them on these files.

## Scripts (`scripts/`)

These are the pipeline scripts, copied unchanged. They hard-code the paths of the session that ran them:
* `J=/home/coder/.claude/jobs/19e53644/tmp` is the work directory, which holds `variants/`, logs and results;
* `S=…/scratchpad` is the directory of `measure_ab2.sh`;
* the build directory is `/home/coder/cuvs/cpp/build/latest`.

To rerun them, point `J` and `S` at a scratch directory that contains `measure_ab2.sh`. The commit hashes in
`build_variants.sh` are those of the 2026-10-09 rebase.

Run order, each step after the previous one finishes, with nothing else building or using the GPU:

1. `build_variants.sh`: builds the base (`staging-test-optimizations-local`), then the base plus each commit alone (cherry-pick;
   a conflict skips that commit). Each build is snapshotted into `$J/variants/<name>/`: the test binaries (hardlinked to the
   base's when identical), the generated `CTestTestfile.cmake` and `libcuvs.so`. Builds use a private local sccache server
   (`SCCACHE_SERVER_PORT=4227`, `SCCACHE_NO_DIST_COMPILE=1`, run through `env -u BASH_ENV` because `/etc/bash.bash_env` resets
   the port).
2. `timing_variants.sh`: for each test executable, runs `measure_ab2.sh` over the base and every variant that touches it, with
   2 interleaved repetitions → `standalone.tsv`.
3. `timing_suite_udf.sh`: FILTER_UDF with a cold and then a warm JIT cache, base vs `udf`, and the whole suite with the base vs
   the per-executable `PERCENT` CTest file (only the generated CTest file is swapped).
4. `post_pipeline.sh` runs the rest:
   * `build_syncfree_chain.sh`: base + `9c688a3c` + `afe7630e` (`sf_pre`), then + `c4ef930b` (`sf_chain`);
   * `build_tip.sh`: the whole branch (`tip`);
   * `timing_syncfree.sh`: IVF-PQ and IVF-Flat, `sf_pre` vs `sf_chain`, 3 repetitions;
   * `recheck_base.sh`: 2 warm-cache runs of the base for every executable;
   * `suite_tip.sh`: whole suite, base + `PERCENT` vs `tip`, alternated after a warm-up run. Test commands are pointed at
     the snapshot, and its `libcuvs.so` is preloaded;
   * `ci_shard_cache.sh`: CI's 4 shard test lists, each with an empty JIT cache and then with its own cache.

`measure_ab2.sh OUTDIR REPS "label=dir ..." EXE...` times whole test executables. In every repetition it runs all variants back
to back, reverses their order every other repetition, and preloads each variant's `libcuvs.so` when the snapshot has one.

`deps_guard.sh` fingerprints the local raft/rmm checkouts and libraries, and stops a script if they change mid-run;
`deps_ref.txt` is the fingerprint recorded for this run. To record a new one, source `deps_guard.sh` and run
`deps_fp > $J/deps_ref.txt`.

Remove `/tmp/cagra_ace_test_disk` and `/tmp/hnsw_ace_disk_test` before test runs: stale directories from earlier runs make the
HNSW ACE disk tests fail.
