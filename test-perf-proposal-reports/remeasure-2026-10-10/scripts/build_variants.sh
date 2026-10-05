#!/bin/bash
# Build base and base + each single commit; snapshot test binaries (hardlinked to base when identical) + libcuvs.so.
J=/home/coder/.claude/jobs/19e53644/tmp
V=$J/variants
B=/home/coder/cuvs/cpp/build/latest
BASE=$(git -C /home/coder/cuvs rev-parse staging-test-optimizations-local)
cd /home/coder/cuvs
. $J/deps_guard.sh
snap() {  # name
  local d=$V/$1; rm -rf $d; mkdir -p $d
  for f in $(grep -oE '/home/coder/cuvs/cpp/build/latest/gtests/[A-Za-z0-9_]+' $B/tests/CTestTestfile.cmake | sort -u); do
    local n=$(basename $f)
    if [ "$1" != base ] && cmp -s $f $V/base/$n; then ln $V/base/$n $d/$n; else cp $f $d/$n; fi
  done
  cp $B/tests/CTestTestfile.cmake $d/CTestTestfile.cmake
  if [ "$1" != base ] && cmp -s $B/libcuvs.so $V/base/libcuvs.so; then ln $V/base/libcuvs.so $d/libcuvs.so; else cp -L $B/libcuvs.so $d/libcuvs.so; fi
}
build() {  # name
  deps_check
  (time env -u BASH_ENV build-cuvs -j16) > $J/build_v_$1.log 2>&1 || { echo "BUILD FAILED $1"; grep -aE "^FAILED|error:" $J/build_v_$1.log | head -5; return 1; }
  echo "built $1 $(tail -1 $J/build_v_$1.log | grep -oE '[0-9:.]+ total')"
}
variant() {  # name sha
  [ -f $V/$1/libcuvs.so ] && { echo "skip $1 (snapshot exists)"; return 0; }
  git checkout -q --detach $BASE || return 1
  if ! git cherry-pick --no-edit $2 > $J/cp_$1.log 2>&1; then
    echo "CHERRY-PICK CONFLICT $1 ($2)"; git cherry-pick --abort; return 1
  fi
  build $1 && snap $1 && echo "snap $1 $(du -sh --apparent-size $V/$1 | cut -f1)"
}
mkdir -p $V
export SCCACHE_SERVER_PORT=4227 SCCACHE_NO_DIST_COMPILE=1 SCCACHE_ERROR_LOG=$J/sccache_private.log
echo "start $(date +%T) raft=$(git -C /home/coder/raft rev-parse --short HEAD) rmm=$(git -C /home/coder/rmm rev-parse --short HEAD)"
[ -f $V/base/libcuvs.so ] && echo "skip base (snapshot exists)" || { git checkout -q --detach $BASE && build base && snap base && echo "snap base"; }
while read name sha; do variant $name $sha; done <<'LIST'
ivfpq_reuse 9c688a3c
cagra_reuse 57b99653
ivfflat_train afe7630e
ivfsq_reuse 7385f3b8
ace_dedupe 4ddf1bc2
bbq_reuse 273d867e
bug_repro 1e65ccaf
helpers ebe83ece
udf 06376826
vamana_test fab92a69
syncfree c4ef930b
percent fe4822c5
graph_core 8ffe8030
cagra_batch 4935488d
kmeans 45220e85
recompute 7d4f0909
pack 822a88a5
batchio 7b177c9f
kvikio 4a701164
hnswhalf a02dd8e5
LIST
deps_check
git checkout -q test-perf-proposals
echo "end $(date +%T)"
