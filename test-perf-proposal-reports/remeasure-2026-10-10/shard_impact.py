#!/usr/bin/env python3
"""Map standalone per-executable savings onto the 4 CI shards (test lists from the nightly run of f751cadd)."""
import csv, collections, os, statistics, sys
J = "/home/coder/.claude/jobs/19e53644/tmp"
shard = {}
for s in range(1, 5):
    for t in open(f"{J}/ci_logs/shard{s}.txt").read().strip().split("|"): shard[t] = s
w = collections.defaultdict(list)
for r in csv.DictReader(open(f"{J}/ab_standalone/results.tsv"), delimiter="\t"):
    if r["variant"] == "base" and r["rep"] == "1": continue  # cold JIT
    w[(r["exe"], r["variant"])].append(float(r["wall_s"]))
if os.path.exists(f"{J}/ab_recheck/results.tsv"):
    for r in csv.DictReader(open(f"{J}/ab_recheck/results.tsv"), delimiter="\t"): w[(r["exe"], "base")].append(float(r["wall_s"]))
sf = collections.defaultdict(list)
if os.path.exists(f"{J}/ab_syncfree/results.tsv"):
    for r in csv.DictReader(open(f"{J}/ab_syncfree/results.tsv"), delimiter="\t"): sf[(r["exe"], r["variant"])].append(float(r["wall_s"]))
# base single-process time per shard (fastest run), from the suite log for executables not timed standalone
import re
base_suite = {}
for l in open(f"{J}/suite_std_base_r2.log", errors="replace"):
    m = re.search(r"Test +#\d+: (\S+) \.+.*?([\d.]+) sec", l)
    if m: base_suite[m.group(1)] = float(m.group(2))
base = {t: min(w[(t, "base")]) if w.get((t, "base")) else base_suite.get(t, 0.0) for t in shard}
tot = collections.Counter()
for t, s in shard.items(): tot[s] += base[t]
print("base per shard (s):", {s: round(v) for s, v in sorted(tot.items())}, "sum", round(sum(tot.values())), "max", round(max(tot.values())))
variants = sorted({v for (_, v) in w if v != "base"})
rows = []
for v in variants:
    d = collections.Counter()
    for (t, vv), ts in w.items():
        if vv == v and t in shard and w.get((t, "base")): d[shard[t]] += min(w[(t, "base")]) - min(ts)
    rows.append((v, d))
d = collections.Counter()
for (t, vv), ts in sf.items():
    if vv == "sf_chain" and sf.get((t, "sf_pre")): d[shard[t]] += min(sf[(t, "sf_pre")]) - min(ts)
if d: rows.append(("syncfree", d))
print("| variant | shard 1 | shard 2 | shard 3 | shard 4 | sum | max shard after |")
print("|---|---|---|---|---|---|---|")
for v, d in sorted(rows, key=lambda x: -sum(x[1].values())):
    after = {s: tot[s] - d[s] for s in range(1, 5)}
    print(f"| {v} | " + " | ".join(f"{d[s]:.0f}" for s in range(1, 5)) + f" | {sum(d.values()):.0f} | {max(after.values()):.0f} (s{max(after, key=after.get)}) |")
