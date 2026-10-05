#!/bin/bash
J=/home/coder/.claude/jobs/19e53644/tmp
$J/build_syncfree_chain.sh > $J/build_syncfree_chain.out 2>&1 || exit 1
$J/build_tip.sh > $J/build_tip.out 2>&1 || exit 2
$J/timing_syncfree.sh > $J/timing_syncfree.out 2>&1 || exit 3
$J/recheck_base.sh > $J/recheck_base.out 2>&1 || exit 4
$J/suite_tip.sh > $J/suite_tip.out 2>&1 || exit 5
$J/ci_shard_cache.sh > $J/ci_shard_cache.out 2>&1 || exit 6
