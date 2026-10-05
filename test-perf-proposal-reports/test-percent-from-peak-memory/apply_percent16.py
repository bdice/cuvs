#!/usr/bin/env python3
"""Set PERCENT in cpp/tests/CMakeLists.txt from gtest_memory_usage.sh output.

PERCENT = peak / 16 GiB, rounded up to a multiple of 5 (minimum 5). ConfigureTest blocks with
RUN_SERIAL or GPUS other than 1 are left unchanged.
Usage: apply_percent16.py CSV CMakeLists.txt [--dry-run]
"""
import math, re, sys
GPU_BYTES = 16 * 2**30
peaks = {}
for line in open(sys.argv[1]):
    name, _, b = line.strip().partition(",")
    if name and b.strip().isdigit():
        peaks[name] = int(b)
def percent(b):
    return min(100, max(5, 5 * math.ceil(b / GPU_BYTES * 100 / 5)))
path = sys.argv[2]
src = open(path).read()
rows = []
def repl(m):
    block = m.group(0)
    name = re.search(r"NAME\s+(\S+)", block).group(1)
    old = re.search(r"PERCENT\s+(\d+)", block)
    old = old.group(1) if old else "-"
    if name not in peaks:
        rows.append((name, None, old, old, "not measured")); return block
    if "RUN_SERIAL" in block or not re.search(r"GPUS\s+1\b", block):
        rows.append((name, peaks[name], old, old, "RUN_SERIAL, unchanged")); return block
    new = percent(peaks[name])
    rows.append((name, peaks[name], old, str(new), ""))
    return re.sub(r"(PERCENT\s+)\d+", lambda p: p.group(1) + str(new), block)
out = re.sub(r"ConfigureTest\(.*?\n\s*\)", repl, src, flags=re.S)
if "--dry-run" not in sys.argv:
    open(path, "w").write(out)
print("| executable | peak RMM memory | PERCENT before | PERCENT after | note |")
print("|---|---|---|---|---|")
for name, b, o, n, note in sorted(rows, key=lambda r: -(r[1] or 0)):
    mem = f"{b / 2**20:,.0f} MiB" if b is not None else "—"
    print(f"| `{name}` | {mem} | {o} | {n} | {note} |")
