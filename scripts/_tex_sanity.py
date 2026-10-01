#!/usr/bin/env python3
"""Structural sanity check for a .tex fragment: balanced environments, resolvable refs,
and tabular rows matching the column spec. No TeX installation needed."""
import collections
import re
import sys

src = open(sys.argv[1], encoding="utf-8").read()

cb = collections.Counter(re.findall(r"\\begin\{(\w+\*?)\}", src))
ce = collections.Counter(re.findall(r"\\end\{(\w+\*?)\}", src))
bad = {k: (cb[k], ce[k]) for k in set(cb) | set(ce) if cb[k] != ce[k]}
print("unbalanced environments:", bad or "none")

labels = set(re.findall(r"\\label\{([^}]+)\}", src))
refs = set(re.findall(r"\\ref\{([^}]+)\}", src))
print("refs with no label:", sorted(refs - labels) or "none")
print("labels never referenced:", sorted(labels - refs) or "none")

for m in re.finditer(r"\\begin\{tabular\}\{@\{\}(.*?)@\{\}\}(.*?)\\end\{tabular\}", src, re.S):
    spec, body = m.group(1), m.group(2)
    ncol = len(re.findall(r"p\{[^}]*\}|[lcr]", spec))
    rows = [r for r in body.split("\\\\") if "&" in r]
    widths = sorted({len(re.findall(r"(?<!\\)&", r)) + 1 for r in rows})
    flag = "" if widths == [ncol] else "   <-- MISMATCH"
    print(f"tabular {ncol} cols; row widths seen {widths}{flag}")

# a stray % starts a comment and silently eats the rest of the line
for i, line in enumerate(src.splitlines(), 1):
    for m in re.finditer(r"(?<!\\)%", line):
        if not line.lstrip().startswith("%"):
            print(f"line {i}: unescaped % mid-line -> {line.strip()[:70]}")
