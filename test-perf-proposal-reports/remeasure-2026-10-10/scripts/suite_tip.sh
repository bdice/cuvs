#!/bin/bash
# Whole suite: base binaries with the per-executable PERCENT values vs the branch tip (which includes them).
# Test commands are pointed at the variant snapshot, and its libcuvs.so is preloaded. Runs alternate A B B A.
J=/home/coder/.claude/jobs/19e53644/tmp
V=$J/variants
B=/home/coder/cuvs/cpp/build/latest
CT=$B/tests/CTestTestfile.cmake
clean() { rm -rf /tmp/cagra_ace_test_disk /tmp/hnsw_ace_disk_test; }
. $J/deps_guard.sh; deps_check
cp $CT $J/CTestTestfile.build_before_suite_tip.cmake
echo "start $(date +%T)"
suite() {  # label variantdir ctestfile kvikio_mode
  sed -E "s#-Dcommand_to_run=$B/gtests/#-Dcommand_to_run=$2/#" $3 > $CT
  clean
  local t0=$(date +%s.%N)
  if [ -n "$4" ]; then export KVIKIO_COMPAT_MODE=$4; else unset KVIKIO_COMPAT_MODE; fi
  (cd $B && LD_PRELOAD=$2/libcuvs.so ctest -j8 --output-on-failure -E cuvs_c_verify_install_headers > $J/suite_tip_$1.log 2>&1)
  local rc=$?
  echo "suite $1 rc=$rc wall=$(echo "$(date +%s.%N) - $t0" | bc) $(grep -a 'tests passed' $J/suite_tip_$1.log)"
}
suite tip_warmup $V/tip $V/tip/CTestTestfile.cmake ON
suite basepct_r1 $V/base $V/percent/CTestTestfile.cmake ""
suite tip_r1 $V/tip $V/tip/CTestTestfile.cmake ON
suite tip_r2 $V/tip $V/tip/CTestTestfile.cmake ON
suite basepct_r2 $V/base $V/percent/CTestTestfile.cmake ""
cp $J/CTestTestfile.build_before_suite_tip.cmake $CT
echo "end $(date +%T)"
