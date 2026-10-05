#!/bin/bash
# Interleaved A/B timing of gtest executables.
#
# Each variant is a directory holding copies of the test executables. If the directory also holds
# a libcuvs.so, it is used via LD_PRELOAD (its SONAME matches, so the build dir copy is not loaded);
# otherwise the executables load libcuvs.so from
# the build dir through an absolute RPATH, so this is only valid when the library is the same for
# all variants). For every repetition and executable, all variants are run back to back, and the
# order of the variants alternates between repetitions. Results go to OUTDIR/results.tsv
# (rep, exe, variant, wall_s, rc, load1, tests, failures, gtest_s) plus one gtest JSON per run.
#
# Usage: measure_ab.sh OUTDIR REPS "label1=dir1 label2=dir2 ..." EXE...
set -uo pipefail
OUT=$1; REPS=$2; VARIANTS=($3); shift 3
mkdir -p "$OUT"
[[ -f "$OUT/results.tsv" ]] || printf 'rep\texe\tvariant\twall_s\trc\tload1\ttests\tfailures\tgtest_s\n' > "$OUT/results.tsv"
cd /home/coder/cuvs/cpp/build/latest/gtests
for rep in $(seq 1 "$REPS"); do
  order=("${VARIANTS[@]}")
  if (( rep % 2 == 0 )); then  # alternate the order of the variants
    order=(); for (( i=${#VARIANTS[@]}-1; i>=0; i-- )); do order+=("${VARIANTS[$i]}"); done
  fi
  for exe in "$@"; do
    for v in "${order[@]}"; do
      label=${v%%=*}; dir=${v#*=}
      [[ -x "$dir/$exe" ]] || { echo "missing $dir/$exe" >&2; continue; }
      load1=$(cut -d' ' -f1 /proc/loadavg)
      json="$OUT/${exe}.${label}.r${rep}.json"
      s=$(date +%s.%N)
      preload=""; [[ -f "$dir/libcuvs.so" ]] && preload="$dir/libcuvs.so"
      LD_PRELOAD="$preload" "$dir/$exe" --gtest_brief=1 --gtest_output="json:$json" > "$OUT/${exe}.${label}.r${rep}.log" 2>&1
      rc=$?
      wall=$(echo "$(date +%s.%N) - $s" | bc)
      read -r tests failures gtest_s < <(python3 -c "
import json, sys
try:
    d = json.load(open('$json')); print(d['tests'], d['failures'] + d.get('errors', 0), d['time'].rstrip('s'))
except Exception:
    print('nan nan nan')")
      printf '%s\t%s\t%s\t%.2f\t%s\t%s\t%s\t%s\t%s\n' "$rep" "$exe" "$label" "$wall" "$rc" "$load1" \
        "$tests" "$failures" "$gtest_s" | tee -a "$OUT/results.tsv"
    done
  done
done
