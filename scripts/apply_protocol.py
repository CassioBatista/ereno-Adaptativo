#!/usr/bin/env python3
"""Apply the frozen dataset screening protocol (docs/DATASET_PROTOCOL.md, tag protocol-v1).

Deterministic counts on the files, per attack type (the finest label of the release); no
model is trained. Releases are read by audit_loaders.py. For each release (or each table,
when a release holds several tables with their own benign class):

  Q1a  rows whose values contradict their column names (an address column without an
       address, a port outside 0-65535, a time column holding neither a number nor a date)
       or that carry no label are misaligned and dropped; a type with > 1 % of its rows
       misaligned is excluded
  Q1b  identical data files filed under different labels: the types involved are excluded
  Q2a  identifiers removed, a type is excluded if > 50 % of its samples have a feature
       vector that also occurs as benign
  Q2b  a type named after a protocol (ARP, ICMP, DNS, HTTP, MQTT, Modbus, UDP, SYN) is
       excluded if <= 50 % of its samples carry that protocol (when the release has an
       indicator of it)
  Q2c  interval annotations: a type is excluded if the counters of its protocol rise above
       the preceding hour (ratio > 2) in <= 50 % of its intervals
  Q2d  period-wide labels: documented rule (a second labelling is used) or attack-labelled
       samples without content (dropped)
  Q3   >= 10,000 benign samples in the same table
  qualifies with >= 2 types left and Q3 met; then one release per domain class by the
  citations of its requested reference received in 2025-2026 (recorded in the protocol)

  python scripts/apply_protocol.py [<release> ...]      (default: all 22)
Out: results/protocol_v1/<release>.txt, results/protocol_v1/<release>.json,
     results/protocol_v1/summary.md (when all releases have a report)
"""
import hashlib
import io
import json
import os
import re
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

OUT = "results/protocol_v1"
MISALIGNED_MAX = 0.01
BENIGN_SHARE_MAX = 0.50
PROTOCOL_MIN = 0.50
VISIBLE_MIN = 0.50
BENIGN_MIN = 10_000
MIN_TYPES = 2

# section 7 of the protocol: domain classes and citations received in 2025-2026 (OpenAlex,
# 2026-11-03); None = not indexed. Releases outside the four classes are out of scope.
CLASS = {
    "ereno": "IEC 61850 substation", "powerduck": "IEC 61850 substation",
    "cicids2017c": "enterprise network",
    "electra": "industrial control", "morris2": "industrial control", "morris3": "industrial control",
    "morris4": "industrial control", "westermo": "industrial control", "hai2305": "industrial control",
    "unsw_iot": "IoT", "ciciot2023": "IoT", "toniot": "IoT", "iot23": "IoT", "xiiotid": "IoT",
    "kitsune": "IoT", "edge_iiotset": "IoT", "cic_iiot2025": "IoT",
    "iomt2024": None, "pdnet_ids26": None, "uav_nidd": None, "uav_hassler": None, "morris1": None,
}
CITES = {"ereno": 38, "powerduck": 14, "cicids2017c": 84, "electra": 38, "morris2": 49, "morris3": 49,
         "morris4": None, "westermo": None, "hai2305": None, "unsw_iot": 38, "ciciot2023": 982,
         "toniot": 418, "iot23": 75, "xiiotid": 153, "kitsune": 20, "edge_iiotset": 778,
         "cic_iiot2025": 32, "iomt2024": 58, "pdnet_ids26": 0, "uav_nidd": 45, "uav_hassler": 82,
         "morris1": 56}
ORDER = ["ereno", "cicids2017c", "electra", "unsw_iot", "ciciot2023", "toniot", "iot23", "xiiotid",
         "kitsune", "edge_iiotset", "uav_nidd", "pdnet_ids26", "uav_hassler", "iomt2024",
         "cic_iiot2025", "morris1", "morris2", "morris3", "morris4", "powerduck", "westermo", "hai2305"]

PROTOS = {"arp": ["arp"], "icmp": ["icmp"], "dns": ["dns"], "http": ["http", "https"],
          "mqtt": ["mqtt"], "modbus": ["modbus", "mbtcp"], "udp": ["udp"], "syn": ["syn"]}
IANA = {"icmp": 1, "udp": 17}


def section(t):
    print(f"\n=== {t} ===")


def tokens(name):
    """'MITM-ArpSpoofing' -> ['mitm', 'arp', 'spoofing']; 'DDoS-SYN_Flood' -> [..., 'syn', ...]."""
    parts = re.split(r"[^A-Za-z0-9]+", str(name))
    out = []
    for p in parts:
        out += re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", p)
    return [t.lower() for t in out]


def named_protocol(typ):
    tk = set(tokens(typ))
    return [p for p, syn in PROTOS.items() if tk & set(syn)]


# ------------------------------------------------------------------------------------- Q1a
IP = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
MAC = re.compile(r"^([0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}$")
EMPTY = {"", "0", "0.0", "nan", "-", "none", "<na>"}
RULES = [  # (name pattern, check)
    (r"^(src|dst|scr|des|s|d|r)[ _]?ips?$|^ip\.(src|dst)(_host)?$|^(orig|resp)_h$|_ip$", "ip"),
    (r"macs?$", "mac"),
    (r"port$", "port"),
    (r"time_?stamp$|^ts$|^date$|^frame\.time$", "time"),
]


def misaligned(df, cols):
    """A row is misaligned when the value in a named column (address, port, time) is not of
    the kind the name promises while most rows of that column are: the row is shifted
    against the header. A column that holds another representation throughout (addresses
    encoded as integers) marks no row. Missing values and lists of addresses ('a/b') pass."""
    bad = np.zeros(len(df), bool)
    used = []
    for c in cols:
        rule = next((r for rx, r in RULES if re.search(rx, c, re.I)), None)
        if rule is None or c not in df:
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            empty = s.isna() | ((s == 0) if rule in ("ip", "mac") else False)
            if rule == "port":
                ok = s.isna() | (s.between(0, 65535) & (s == s.round()))
            elif rule == "time":
                ok = pd.Series(True, index=s.index)
            else:                                   # an address held as a number
                ok = empty
        else:
            v = s.astype(object).where(s.notna(), "").astype(str).str.strip()
            low = v.str.lower()
            empty = low.isin(EMPTY)
            first = v.str.split(r"[/;, ]+", regex=True).str[0]
            if rule == "ip":
                ok = low.isin(EMPTY) | first.str.match(IP)
            elif rule == "mac":
                ok = low.isin(EMPTY) | first.str.match(MAC) | first.str.match(r"^[0-9a-fA-F]{12}$")
            elif rule == "port":
                x = pd.to_numeric(v, errors="coerce")
                ok = low.isin(EMPTY) | (x.between(0, 65535) & (x == x.round()))
            else:
                x = pd.to_numeric(v, errors="coerce")
                d = pd.to_datetime(v, errors="coerce", format="mixed")
                ok = low.isin(EMPTY) | x.notna() | d.notna()
        ok = ok.values
        nonempty = ~np.asarray(empty, bool)
        if nonempty.any() and ok[nonempty].mean() < 0.5:
            used.append(f"{c} (holds another representation throughout: not checked)")
            continue
        bad |= ~ok
        used.append(c)
    return bad, used


# ------------------------------------------------------------------------------------- Q2b
def carries(df, proto, cols):
    """Rows that carry the protocol, from columns named after it (non-zero), or from an IP
    protocol field (IANA number or name). None when the release has no indicator."""
    named = [c for c in cols if set(tokens(c)) & set(PROTOS[proto]) and c in df]
    pf = next((c for c in cols if c.lower() in ("protocol", "proto", "ip.proto") and c in df), None)
    hit, have = np.zeros(len(df), bool), False
    for c in named:
        s = df[c]
        v = (pd.to_numeric(s, errors="coerce").fillna(0) != 0) if pd.api.types.is_numeric_dtype(s) \
            else ~s.astype(str).str.strip().str.lower().isin(EMPTY)
        hit |= v.values
        have = True
    if pf is not None and proto in ("icmp", "udp", "arp"):
        s = df[pf]
        x = pd.to_numeric(s, errors="coerce")
        if proto == "arp":
            have = True        # an IP protocol field: an ARP frame is not an IP flow
        else:
            hit |= ((x == IANA[proto]) | (s.astype(str).str.lower() == proto)).values
            have = True
    return hit if have else None


# ------------------------------------------------------------------------------------- Q1b
def file_duplicates(name, rel, types):
    """Types filed in the same data file as another label (checksums)."""
    out = {}
    if rel.labelled_files:
        md5 = {}
        for p, lab in rel.labelled_files:
            h = hashlib.md5()
            with open(p, "rb") as f:
                for blk in iter(lambda: f.read(1 << 22), b""):
                    h.update(blk)
            md5.setdefault(h.hexdigest(), set()).add(lab)
        for labs in md5.values():
            if len(labs) > 1:
                for lab in labs:
                    out[lab] = f"same file as {sorted(labs - {lab})}"
    if name.startswith("uav_nidd"):
        # raw release listing (scripts/list_uav_nidd_figshare.py): files with the same CRC32
        # and size in folders of different attacks; folder -> label by the first four letters
        txt = open("results/uav_nidd_figshare_listing.txt").read()
        block = txt.split("same CRC32", 1)[-1]
        groups, cur = [], None
        for line in block.splitlines():
            if re.match(r"^\s+[\d,]+\s+crc=", line):
                cur = []
                groups.append(cur)
            elif cur is not None and "/" in line:
                cur.append(line.strip())
        key = {"case1_uav": "UAV-Case 1", "case2_ap": "Access Point Case2", "case3_gcs": "GCS Case3"}

        def norm(s):
            s = re.sub(r"(?i)\b(ap|uav|gcs|packets?|attack)\b|-ap$|-uav$", " ", s)
            return re.sub(r"[^a-z]", "", s.lower())[:4]
        case = next(k for k in key if k in rel.name.replace(", ", "_").replace("UAV-NIDD_", ""))
        lab4 = {norm(t): t for t in types}
        for g in groups:
            folders = {(p.split("/")[0], norm(p.split("/")[1])) for p in g}
            if len({f for _, f in folders}) < 2:
                continue
            for c, f in folders:
                if c == key[case] and f in lab4:
                    others = sorted({o for cc, o in folders if o != f})
                    out[lab4[f]] = f"raw capture identical to folders {others}"
    return out


# ------------------------------------------------------------------------------------- Q2c
def unsw_visibility():
    import inspect_unsw_iot as U
    X, A = U.load()
    feats = [c for c in X.columns if not c.startswith("_")]
    res = []
    for _, r in A.iterrows():
        fam = U.FAMILY.get(r["cat"])
        g = X[X["_dev"] == r["_dev"]]
        cols = [c for c in feats if fam in c and c.endswith("Packet")]
        ins = g[(g["_ts"] >= r["start"]) & (g["_ts"] < r["end"])][cols].sum(axis=1)
        pre = g[(g["_ts"] >= r["start"] - 3600) & (g["_ts"] < r["start"] - 60)][cols].sum(axis=1)
        if len(ins) and len(pre):
            res.append((r["cat"], (ins.mean() + 1) / (pre.mean() + 1) > 2))
    R = pd.DataFrame(res, columns=["cat", "visible"])
    return R.groupby("cat")["visible"].agg(["mean", "size"])


# ------------------------------------------------------------------------------------- run
def screen(name, rel):
    import audit_protocol as P
    df = rel.df
    typ = (df["_type"] if "_type" in df else df["_y"]).astype(str).values
    benign = rel.benign
    if "_type" in df:
        # the benign class is the release's own benign label (the most frequent finest label
        # under the loader's benign category); other labels the loader folded into benign,
        # such as the 'Attempted' flows of CICIDS2017, stay attack types of their own
        yb = df["_y"].astype(str).values == benign
        btype = pd.Series(typ[yb]).mode().iloc[0] if yb.any() else benign
        typ = np.where(typ == btype, benign, typ)
    res = {"release": rel.name, "rows": len(df), "sample": rel.sample, "types": {}}
    print(f"SCREENING (protocol-v1) — {rel.name}")
    print(f"rows {len(df):,} ({rel.sample}); benign label '{benign}'; content features "
          f"{len(rel.feats)}; identifiers {len(rel.ids)}")
    types = [t for t in pd.Series(typ).value_counts().index if t != benign]
    dec = {t: [] for t in types}

    section("Q1a  misaligned rows")
    unl = pd.Series(typ).str.lower().isin(["nan", "none", ""]).values
    bad, used = misaligned(df, rel.feats + rel.ids)
    bad |= unl
    print(f"columns checked: {used or 'none (no address, port or time column)'}; rows without "
          f"a label {int(unl.sum()):,}; misaligned rows {int(bad.sum()):,} ({100 * bad.mean():.2f}%)")
    for t in types:
        m = typ == t
        share = bad[m].mean()
        if share > MISALIGNED_MAX:
            dec[t].append(f"Q1a {100 * share:.2f}% misaligned")
        print(f"  {t:<44} rows {int(m.sum()):>10,}  misaligned {100 * share:6.2f}%")
    nb = int(bad[typ == benign].sum())
    print(f"  {benign:<44} rows {int((typ == benign).sum()):>10,}  misaligned "
          f"{100 * bad[typ == benign].mean() if (typ == benign).any() else 0:6.2f}%")
    keep = ~bad

    section("Q1b  identical files under different labels")
    dup = file_duplicates(name, rel, types)
    for t, why in dup.items():
        if t in dec:
            dec[t].append(f"Q1b {why}")
    print("  " + ("; ".join(f"{t}: {w}" for t, w in dup.items()) if dup else "none found"
                  + ("" if rel.labelled_files or name.startswith("uav_nidd") else
                     " (the release has no per-label data files)")))

    section("Q2d  period-wide labels")
    empty = np.zeros(len(df), bool)
    if name.startswith("cic_iiot2025"):
        num = df[rel.feats].select_dtypes("number")
        empty = (num.fillna(0) == 0).all(axis=1).values & (typ != benign)
        print(f"  attack-labelled windows with no traffic and no sensor reading: {int(empty.sum()):,} "
              "(dropped)")
    elif name == "westermo_1":
        for t in types:
            dec[t].append("Q2d timing labelling marks every flow during an event; the release's "
                          "attacker-involvement labelling is used instead")
        print("  timing labelling (IT): documented as every flow during an event -> not used")
    else:
        print("  no documented period-wide rule and no attack-labelled sample without content")
    keep &= ~empty

    section("Q2a  samples whose content vector also occurs as benign")
    d = df.loc[keep]
    ty = typ[keep]
    h = P.vec_hash(d, rel.feats)
    ben = np.unique(h[ty == benign])
    for t in types:
        m = ty == t
        if not m.any():
            dec[t].append("no rows left")
            continue
        share = np.isin(h[m], ben).mean()
        if share > BENIGN_SHARE_MAX:
            dec[t].append(f"Q2a {100 * share:.1f}% identical to benign")
        print(f"  {t:<44} {100 * share:6.2f}%")

    section("Q2b  protocol named by the type")
    for t in types:
        for p in named_protocol(t):
            m = ty == t
            c = carries(d.loc[m] if m.any() else d.iloc[:0], p, rel.feats + rel.ids)
            if c is None:
                print(f"  {t:<44} {p.upper():<6} no indicator in the release: not applied")
                continue
            share = c.mean() if len(c) else 0.0
            if share <= PROTOCOL_MIN:
                dec[t].append(f"Q2b {100 * share:.1f}% carry {p.upper()}")
            print(f"  {t:<44} {p.upper():<6} carried by {100 * share:6.2f}%")

    section("Q2c  annotation visible in the counters")
    if name == "unsw_iot":
        vis = unsw_visibility()
        for t in types:
            if t in vis.index:
                v = vis.loc[t, "mean"]
                if v <= VISIBLE_MIN:
                    dec[t].append(f"Q2c visible in {100 * v:.0f}% of intervals")
                print(f"  {t:<44} visible in {100 * v:5.1f}% of {int(vis.loc[t, 'size'])} intervals")
    else:
        print("  labels are per sample: not applicable")

    section("Q3  benign samples")
    nben = int((ty == benign).sum())
    q3 = nben >= BENIGN_MIN
    print(f"  {nben:,} benign samples after Q1-Q2d ({'meets' if q3 else 'below'} {BENIGN_MIN:,})")

    section("DECISION")
    left = [t for t in types if not dec[t]]
    for t in types:
        print(f"  {t:<44} {'kept' if not dec[t] else 'excluded: ' + '; '.join(dec[t])}")
        res["types"][t] = {"rows": int((typ == t).sum()), "excluded_by": dec[t]}
    qual = q3 and len(left) >= MIN_TYPES
    print(f"\n  types kept {len(left)} of {len(types)}; benign {nben:,} -> "
          f"{'QUALIFIES' if qual else 'DOES NOT QUALIFY'}")
    res.update(kept=left, benign=nben, q3=q3, qualifies=qual)
    return res


def run(name):
    import audit_loaders as L
    rels = L.LOADERS[name]()
    rels = rels if isinstance(rels, list) else [rels]
    out = []
    os.makedirs(OUT, exist_ok=True)
    for i, rel in enumerate(rels):
        sub = name if len(rels) == 1 else f"{name}_{i + 1}"
        buf = io.StringIO()
        with redirect_stdout(buf):
            r = screen(sub, rel)
        r["table"] = sub
        open(f"{OUT}/{sub}.txt", "w").write(buf.getvalue())
        print(buf.getvalue())
        out.append(r)
    best = max(out, key=lambda r: (r["qualifies"], len(r["kept"]), r["benign"]))
    json.dump({"release": name, "tables": out, "qualifies": best["qualifies"], "best_table": best["table"]},
              open(f"{OUT}/{name}.json", "w"), indent=1)


def summary():
    rows, by_class = [], {}
    for n in ORDER:
        p = f"{OUT}/{n}.json"
        if not os.path.exists(p):
            return print("summary skipped: missing", n)
        j = json.load(open(p))
        best = next(t for t in j["tables"] if t["table"] == j["best_table"])
        rows.append((n, j, best))
        if j["qualifies"] and CLASS[n]:
            by_class.setdefault(CLASS[n], []).append(n)
    sel = {}
    for cl, ns in by_class.items():
        if "ereno" in ns:
            sel[cl] = "ereno"
            continue
        ben = {n: next(t for t in json.load(open(f"{OUT}/{n}.json"))["tables"]
                       if t["table"] == json.load(open(f"{OUT}/{n}.json"))["best_table"])["benign"] for n in ns}
        sel[cl] = max(ns, key=lambda n: (CITES[n] if CITES[n] is not None else -1, ben[n]))
    L = ["# Dataset screening — protocol-v1", "",
         "Applied by `scripts/apply_protocol.py`; per-release reports in this folder.", "",
         "| Release | Class | Types kept / total | Benign | Qualifies | Citations 2025-26 | Selected |",
         "|---|---|---|---|---|---|---|"]
    for n, j, b in rows:
        tot = len(b["types"])
        L.append(f"| {n} ({b['table']}) | {CLASS[n] or 'out of scope'} | {len(b['kept'])} / {tot} | "
                 f"{b['benign']:,} | {'yes' if j['qualifies'] else 'no'} | "
                 f"{CITES[n] if CITES[n] is not None else 'n.i.'} | {'**yes**' if sel.get(CLASS[n]) == n else ''} |")
    L += ["", "Excluded types per release:", ""]
    for n, j, b in rows:
        ex = {t: v["excluded_by"] for t, v in b["types"].items() if v["excluded_by"]}
        if ex:
            L.append(f"- **{n}**: " + "; ".join(f"{t} ({', '.join(v)})" for t, v in ex.items()))
    open(f"{OUT}/summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    os.chdir(os.path.expanduser("~/ereno-Adaptativo"))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    names = sys.argv[1:] or ORDER
    for n in names:
        if n != "--summary":
            run(n)
    summary()
