#!/bin/bash
# Mimic CI's 4 test shards (test lists from the nightly run of f751cadd) on the base build with base PERCENT values:
# each shard first with an empty JIT cache (today, shards 1, 3 and 4 never save their cache), then again with the cache
# it just filled (per-shard cache key, c8e99eb3).
J=/home/coder/.claude/jobs/19e53644/tmp
V=$J/variants
B=/home/coder/cuvs/cpp/build/latest
CT=$B/tests/CTestTestfile.cmake
. $J/deps_guard.sh; deps_check
cp $CT $J/CTestTestfile.build_before_ci_shard.cmake
sed -E "s#-Dcommand_to_run=$B/gtests/#-Dcommand_to_run=$V/base/#" $V/base/CTestTestfile.cmake > $CT
echo "start $(date +%T)"
for s in 1 2 3 4; do
  d=$(mktemp -d -p $J jitcache_shard$s.XXXX)
  for w in cold warm; do
    rm -rf /tmp/cagra_ace_test_disk /tmp/hnsw_ace_disk_test
    t0=$(date +%s.%N)
    (cd $B && CUDA_CACHE_PATH=$d LD_PRELOAD=$V/base/libcuvs.so ctest -j8 --output-on-failure -R "^($(cat $J/ci_logs/shard$s.txt))\$" > $J/ci_shard${s}_$w.log 2>&1)
    echo "shard $s $w rc=$? wall=$(echo "$(date +%s.%N) - $t0" | bc) $(grep -a 'tests passed' $J/ci_shard${s}_$w.log) cache=$(du -sh $d | cut -f1)"
  done
  rm -rf $d
done
cp $J/CTestTestfile.build_before_ci_shard.cmake $CT
echo "end $(date +%T)"
