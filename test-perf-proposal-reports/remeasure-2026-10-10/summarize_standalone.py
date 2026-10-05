#!/usr/bin/env python3
"""Per-commit standalone savings from ab_standalone/results.tsv (base vs base + one commit)."""
import csv, collections, statistics, sys
# Estimator: the fastest run of each (executable, variant). A variant's first run can pay cold JIT linking for kernels
# its libcuvs changes (e.g. IVF-PQ with 7d4f0909: 455 then 396 s), which only ever adds time.
EST = min if "--mean" not in sys.argv else statistics.mean
J = "/home/coder/.claude/jobs/19e53644/tmp"
rows = list(csv.DictReader(open(f"{J}/ab_standalone/results.tsv"), delimiter="\t"))
w = collections.defaultdict(list); bad = []
for r in rows:
    w[(r["exe"], r["variant"])].append(float(r["wall_s"]))
    if r["rc"] != "0" or r["failures"] not in ("0",):
        bad.append((r["exe"], r["variant"], r["rep"], r["rc"], r["failures"]))
info = {
 "ivfpq_reuse": ("9c688a3c", "IVF-PQ test index reuse + dedupe", "test", "+170/−41"),
 "cagra_reuse": ("57b99653", "CAGRA test index reuse", "test", "+591/−285"),
 "ivfflat_train": ("afe7630e", "IVF-Flat trains each index once", "test", "+331/−141"),
 "ivfsq_reuse": ("7385f3b8", "IVF-SQ quantizer/index reuse", "test", "+137/−19"),
 "ace_dedupe": ("4ddf1bc2", "HNSW-ACE npartitions dedupe", "test", "+10/−2"),
 "bbq_reuse": ("273d867e", "BBQ graph/reference reuse", "test", "+139/−22"),
 "bug_repro": ("1e65ccaf", "cheaper CAGRA bug reproducers", "test", "+50/−16"),
 "helpers": ("ebe83ece", "faster calc_recall/check_unique_indices", "test", "+143/−52"),
 "udf": ("06376826", "one parameterized filter UDF", "test", "+27/−30"),
 "vamana_test": ("fab92a69", "Vamana degree-32 sweep at one batch size", "test", "+17/−1"),
 "syncfree": ("c4ef930b", "sync-free IVF-Flat/IVF-PQ checks", "test", "+151/−80"),
 "graph_core": ("8ffe8030", "CAGRA reverse graph from one device copy", "library", "+75/−19"),
 "cagra_batch": ("4935488d", "CAGRA batched search for unaligned dims", "library", "+50/−67"),
 "kmeans": ("45220e85", "balanced k-means per-iteration overhead", "library", "+546/−172"),
 "recompute": ("7d4f0909", "IVF list pointers in one copy", "library", "+11/−6"),
 "pack": ("822a88a5", "IVF-Flat pack/unpack kernels", "library", "+85/−46"),
 "batchio": ("7b177c9f", "batched IVF list I/O", "library", "+423/−31"),
 "kvikio": ("4a701164", "honour KVIKIO_COMPAT_MODE (+ CI env)", "library+CI", "+25/−2"),
 "hnswhalf": ("a02dd8e5", "hnswlib half distance in float", "library", "+239/−2"),
}
# Base rep 1 was each group's first run and paid cold JIT linking for the rebased libcuvs; use base rep 2 plus the
# warm-cache recheck (ab_recheck) instead.
import os
for k in [k for k in w if k[1] == "base"]: w[k] = []
for r in rows:
    if r["variant"] == "base" and r["rep"] == "2": w[(r["exe"], "base")].append(float(r["wall_s"]))
rc = f"{J}/ab_recheck/results.tsv"
if os.path.exists(rc):
    for r in csv.DictReader(open(rc), delimiter="\t"):
        w[(r["exe"], "base")].append(float(r["wall_s"]))
        if r["rc"] != "0" or r["failures"] != "0": bad.append((r["exe"], "base-recheck", r["rep"], r["rc"], r["failures"]))
# sync-free IVF checks only apply on top of the IVF-PQ / IVF-Flat test reuse commits: chain sf_pre -> sf_chain.
sf = f"{J}/ab_syncfree/results.tsv"
sfw = collections.defaultdict(list)
if os.path.exists(sf):
    for r in csv.DictReader(open(sf), delimiter="\t"):
        sfw[(r["exe"], r["variant"])].append(float(r["wall_s"]))
        if r["rc"] != "0" or r["failures"] != "0": bad.append((r["exe"], r["variant"], r["rep"], r["rc"], r["failures"]))
per = collections.defaultdict(list)
for (exe, v), ts in sfw.items():
    if v == "sf_chain" and sfw.get((exe, "sf_pre")):
        b = sfw[(exe, "sf_pre")]
        per["syncfree"].append((exe, EST(b), EST(ts), min(b), max(b), min(ts), max(ts)))
for (exe, v), ts in w.items():
    if v == "base": continue
    b = w.get((exe, "base"))
    if not b: continue
    per[v].append((exe, EST(b), EST(ts), min(b), max(b), min(ts), max(ts)))
out = []
for v, lst in per.items():
    tot = sum(x[1] - x[2] for x in lst)
    out.append((tot, v, lst))
out.sort(reverse=True)
print("| commit | change | kind | lines | saved (s) | where |")
print("|---|---|---|---|---|---|")
for tot, v, lst in out:
    sha, desc, kind, lines = info.get(v, ("?", v, "?", "?"))
    where = ", ".join(f"{e.replace('NEIGHBORS_ANN_','').replace('NEIGHBORS_','').replace('_UINT32_TEST','').replace('_TEST','')} {b:.0f}→{a:.0f}" for e, b, a, *_ in sorted(lst, key=lambda x: x[2]-x[1]) if abs(b-a) >= 0.3)
    print(f"| {sha} | {desc} | {kind} | {lines} | {tot:.1f} | {where} |")
if bad: print("FAILURES:", bad)
