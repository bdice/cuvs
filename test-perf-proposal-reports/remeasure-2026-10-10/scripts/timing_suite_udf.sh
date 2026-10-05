#!/bin/bash
# Whole-suite PERCENT comparison and FILTER_UDF cold-cache comparison on the base build.
J=/home/coder/.claude/jobs/19e53644/tmp
V=$J/variants
B=/home/coder/cuvs/cpp/build/latest
CT=$B/tests/CTestTestfile.cmake
clean() { rm -rf /tmp/cagra_ace_test_disk /tmp/hnsw_ace_disk_test; }
. $J/deps_guard.sh; deps_check
echo "start $(date +%T)"
cd /home/coder/cuvs && git checkout -q --detach staging-test-optimizations-local && (SCCACHE_SERVER_PORT=4227 SCCACHE_NO_DIST_COMPILE=1 SCCACHE_ERROR_LOG=$J/sccache_private.log env -u BASH_ENV build-cuvs -j16 > $J/build_suite_base.log 2>&1) || { echo "base build failed"; exit 1; }
cmp -s $B/libcuvs.so $V/base/libcuvs.so && echo "build dir libcuvs.so == base snapshot" || echo "WARNING: build dir libcuvs.so differs from base snapshot"
cp $CT $J/CTestTestfile.base_now.cmake
diff <(sed -E 's/gpus:[0-9]+/gpus:N/g' $V/base/CTestTestfile.cmake) <(sed -E 's/gpus:[0-9]+/gpus:N/g' $V/percent/CTestTestfile.cmake) > /dev/null && echo "base/percent CTest files differ only in PERCENT"
# FILTER_UDF: fresh JIT cache per run (cold), then the same cache again (warm); alternate base / udf
for order in "base udf" "udf base"; do for v in $order; do
  d=$(mktemp -d -p $J)
  for w in cold warm; do
    t0=$(date +%s.%N); (cd $B/gtests && CUDA_CACHE_PATH=$d LD_PRELOAD=$V/$v/libcuvs.so $V/$v/NEIGHBORS_ANN_CAGRA_FILTER_UDF_TEST --gtest_brief=1 > $J/udf_${v}_${w}.log 2>&1); rc=$?
    echo "udf $v $w rc=$rc wall=$(echo "$(date +%s.%N) - $t0" | bc)"
  done; rm -rf $d
done; done
suite() {  # label ctestfile
  cp $2 $CT; clean
  local t0=$(date +%s.%N)
  (cd $B && ctest -j8 --output-on-failure -E cuvs_c_verify_install_headers > $J/suite_std_$1.log 2>&1)
  local rc=$?
  echo "suite $1 rc=$rc wall=$(echo "$(date +%s.%N) - $t0" | bc) $(grep -a 'tests passed' $J/suite_std_$1.log)"
}
suite base_r1 $V/base/CTestTestfile.cmake
suite percent_r1 $V/percent/CTestTestfile.cmake
suite percent_r2 $V/percent/CTestTestfile.cmake
suite base_r2 $V/base/CTestTestfile.cmake
cp $V/base/CTestTestfile.cmake $CT
git -C /home/coder/cuvs checkout -q test-perf-proposals
echo "end $(date +%T)"
