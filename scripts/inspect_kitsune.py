#!/usr/bin/env python3
"""Audit and evaluation protocol on Kitsune (Mirsky et al., NDSS 2018): IP camera and IoT
networks, one capture per attack, per-packet labels, 115 AfterImage features (incremental
statistics per source MAC-IP, source IP, channel and socket over five decay windows).
Copy: the first author's Kaggle upload ymirsky/network-attack-dataset-kitsune, per-file
(checksums in results/kitsune_md5sums.txt). Five attacks whose labels are not path labels:
Mirai, OS Scan, Fuzzing, SYN DoS, SSDP Flood (ARP MitM, Active Wiretap, Video Injection and
SSL Renegotiation label every packet that crossed the man in the middle, or are not used).

Rows are read in file (capture) order with a uniform stride, at most CAP rows per capture.

  (i)   STRUCTURE   rows per capture, labels aligned with features, NaN/inf
  (ii)  LABELS      onset of the attack; benign share after onset; label runs;
                    duplicates and vectors carrying both labels
  (iii) TIME        benign traffic before and interleaved after the onset
  (iv)  ATTRIBUTION no addresses in the feature files (pcaps only)
  (v)   PREVALENCE  malicious share per capture
  (vi)  SHORTCUT    can NORMAL packets be assigned to their capture? (session identity)
Then the ReSIDS protocol: 10 specialists (two per attack), normal traffic pooled from the
five captures, time-ordered split inside each capture (first 70 % train).

  python scripts/inspect_kitsune.py
Out: results/kitsune_audit.txt
"""
import glob
import io
import os
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cross_domain_results as C  # noqa: E402

D = os.path.expanduser("~/datasets/kitsune")
CAPS = {"Mirai": ("Mirai_dataset.csv", "mirai_labels.csv"),
        "OS_Scan": ("OS_Scan_dataset.csv", "OS_Scan_labels.csv"),
        "Fuzzing": ("Fuzzing_dataset.csv", "Fuzzing_labels.csv"),
        "SYN_DoS": ("SYN_DoS_dataset.csv", "SYN_DoS_labels.csv"),
        "SSDP_Flood": ("SSDP_Flood_dataset.csv", "SSDP_Flood_labels.csv")}
CAP = 400_000
# AfterImage feature order: MI_dir 0-14, H 15-29, HH 30-64, HH_jit 65-79, HpHp 80-114.
# The jitter statistics hold the raw packet timestamp (mean) and its square (variance)
# whenever a channel is new, i.e. the absolute capture time: a session clock.
JIT = list(range(65, 80))


def find(name):
    hits = glob.glob(os.path.join(D, "**", name), recursive=True)
    return hits[0]


def read_labels(path):
    """One label per packet, last column; files with and without a header line exist."""
    lab = pd.read_csv(path, header=None, dtype=str)
    col = lab.iloc[:, -1]
    if not str(col.iloc[0]).strip().lstrip("-").replace(".", "").isdigit():
        col = col.iloc[1:]
    return pd.to_numeric(col).astype(int).values


def load():
    """Stride sample of each capture, cached (reading the 33 GB of CSV takes long)."""
    cache = os.path.join(D, "sample_cache.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        print("loaded sample cache", file=sys.stderr)
        return list(z["parts"])
    parts = _load()
    np.savez(cache, parts=np.array(parts, dtype=object))
    return parts


def _load():
    parts = []
    for cap, (fd, fl) in CAPS.items():
        y = read_labels(find(fl))
        n = len(y)
        step = max(1, int(np.ceil(n / CAP)))
        rows = []
        for k, ch in enumerate(pd.read_csv(find(fd), header=None, chunksize=500_000, dtype=np.float32)):
            start = k * 500_000
            sel = np.arange(start, start + len(ch))
            keep = (sel % step) == 0
            rows.append(ch.values[keep])
        Xc = np.vstack(rows)
        if Xc.shape[1] == 116:            # leading index column in some uploads
            Xc = Xc[:, 1:]
        idx = np.arange(0, n, step)[: len(Xc)]
        parts.append((cap, Xc, y[idx], idx, n, y))
        print(f"loaded {cap}: rows {n:,}, kept {len(Xc):,} (stride {step}), features {Xc.shape[1]}",
              file=sys.stderr, flush=True)
    return parts


def main():
    parts = load()
    print("=== (i) STRUCTURE / (ii) LABELS / (iii) TIME / (v) PREVALENCE ===")
    for cap, Xc, yc, idx, n, yfull in parts:
        onset = int(np.argmax(yfull == 1)) if (yfull == 1).any() else -1
        after = yfull[onset:] if onset >= 0 else yfull[:0]
        runs = int((np.diff(yfull) != 0).sum()) + 1
        bad = int((~np.isfinite(Xc)).sum())
        print(f"  {cap:<11} packets {n:>10,}  malicious {pct(int(yfull.sum()), n)}  onset at packet "
              f"{onset:,} ({pct(onset, n)})  benign share after onset {pct(int((after == 0).sum()), len(after))}  "
              f"label runs {runs:,}  non-finite values in sample {bad}")
    print("\nduplicates and vectors with both labels (per capture sample):")
    for cap, Xc, yc, *_ in parts:
        h = pd.util.hash_pandas_object(pd.DataFrame(Xc), index=False)
        pairs = pd.DataFrame({"h": h.values, "y": yc}).drop_duplicates()
        multi = pairs["h"][pairs["h"].duplicated()].unique()
        print(f"  {cap:<11} duplicates {pct(int(h.duplicated().sum()), len(h))}; rows in vectors with both "
              f"labels {pct(int(np.isin(h.values, multi).sum()), len(h))}")
    print("\n=== (iv) ATTRIBUTION === the feature files carry no addresses; attribution needs the pcaps")

    print("\n=== timestamp leak: share of rows whose jitter columns hold an epoch-like value (> 1e8) ===")
    for cap, Xc, *_ in parts:
        print(f"  {cap:<11} {pct(int((np.abs(Xc[:, JIT]) > 1e8).any(1).sum()), len(Xc))}")
    keep_all = list(range(parts[0][1].shape[1]))
    keep_nojit = [i for i in keep_all if i not in JIT]
    for tag, cols in (("all 115 features", keep_all), ("without the 15 jitter features", keep_nojit)):
        probe_and_protocol(parts, cols, tag)


def probe_and_protocol(parts, cols, tag):
    print(f"\n################ {tag} ################")
    parts = [(cap, Xc[:, cols], yc, *rest) for cap, Xc, yc, *rest in parts]
    print("\n=== (vi) SHORTCUT PROBE: assign NORMAL packets to their capture (5 classes) ===")
    from sklearn.metrics import f1_score
    Xn, cn, on = [], [], []
    for k, (cap, Xc, yc, *_ ) in enumerate(parts):
        b = np.where(yc == 0)[0]
        Xn.append(Xc[b]); cn.append(np.full(len(b), k)); on.append(np.arange(len(b)) / max(1, len(b)))
    Xn, cn, on = np.vstack(Xn), np.concatenate(cn), np.concatenate(on)
    rng = np.random.RandomState(42)
    sel = np.concatenate([rng.choice(np.where(cn == k)[0], min(50_000, int((cn == k).sum())), replace=False)
                          for k in range(len(parts))])
    tr = sel[on[sel] < 0.7]; te = sel[on[sel] >= 0.7]
    prm = {"objective": "multi:softmax", "num_class": len(parts), "max_depth": 6, "eta": 0.3,
           "tree_method": "hist", "nthread": 4, "seed": 42}
    m = xgb.train(prm, xgb.DMatrix(Xn[tr], label=cn[tr]), num_boost_round=50)
    p = m.predict(xgb.DMatrix(Xn[te])).astype(int)
    per = f1_score(cn[te], p, average=None, zero_division=0)
    print(f"  time-ordered within each capture: macro-F1 {per.mean():.3f}  "
          + "  ".join(f"{parts[k][0]}={v:.2f}" for k, v in enumerate(per)))
    print("  (near 1.0 = normal traffic identifies its capture: a specialist can learn the capture)")

    print("\n=== ReSIDS PROTOCOL (10 specialists, time-ordered split inside each capture) ===")
    cats = [c for c, *_ in parts]
    Xs, ys, cs, trm = [], [], [], []
    for cap, Xc, yc, *_ in parts:
        # time order inside each capture, per class: the first 70 % of its normal packets and
        # the first 70 % of its attack packets train (a plain 70 % cut of the capture leaves
        # no attack in training where the onset is late: OS Scan 77 %, SYN DoS 97 %)
        tr = np.zeros(len(yc), bool)
        for lab in (0, 1):
            ii = np.where(yc == lab)[0]
            tr[ii[: int(0.7 * len(ii))]] = True
        Xs.append(Xc); ys.append(yc); cs.append(np.where(yc == 1, cap, "Normal"))
        trm.append(tr)
    X, cat, trm = np.vstack(Xs), np.concatenate(cs), np.concatenate(trm)
    capid = np.concatenate([np.full(len(p[2]), p[0]) for p in parts])
    print("train:", pd.Series(cat[trm]).value_counts().to_dict())
    print("test: ", pd.Series(cat[~trm]).value_counts().to_dict())
    rng = np.random.default_rng(C.SEED)
    ben = np.where(cat[trm] == "Normal")[0]
    N = 2 * len(cats)
    shares = np.array_split(rng.permutation(ben), N)
    halves = {c: np.array_split(rng.permutation(np.where(cat[trm] == c)[0]), 2) for c in cats}
    Xtr = X[trm]
    models = []
    for i in range(N):
        c = cats[i % len(cats)]
        pos, neg = halves[c][i // len(cats)], shares[i]
        rows = np.r_[pos, neg]
        q = dict(C.XGB); q["scale_pos_weight"] = len(neg) / max(len(pos), 1)
        models.append(xgb.train(q, xgb.DMatrix(Xtr[rows], label=np.r_[np.ones(len(pos)), np.zeros(len(neg))]),
                                num_boost_round=10))
    Xte, cte, kte = X[~trm], cat[~trm], capid[~trm]
    V = np.column_stack([(mm.predict(xgb.DMatrix(Xte)) >= 0.5).astype(np.int8) for mm in models])
    atk = cte != "Normal"
    for k in (1, 2):
        g = C.stats(V.sum(1) >= k, atk)
        print(f"  GL (union of {N}) k>={k}: recall {g['recall']:.2f}%  FPR {g['FPR']:.3f}%  "
              f"F1 {g['F1']:.2f}  F1@13.75 {g['F1@13.75']:.2f}")
    f2 = V.sum(1) >= 2
    print("  per attack recall (GL k>=2):", {c: round(100 * f2[cte == c].mean(), 1) for c in cats})
    print("  FPR on the normal packets of each capture (GL k>=2):",
          {c: round(100 * f2[(~atk) & (kte == c)].mean(), 3) for c in cats})
    for nn in (len(cats), 3):
        v = V[:, :nn].sum(1) >= 2
        print(f"  FL k>=2 with {nn} nodes: recall {100 * v[atk].mean():.1f}%  FPR {100 * v[~atk].mean():.3f}%  "
              + str({c: round(100 * v[cte == c].mean(), 1) for c in cats}))


def pct(a, b):
    return f"{100 * a / b:.2f}%" if b else "n/a"


if __name__ == "__main__":
    os.chdir(os.path.expanduser("~/ereno-Adaptativo"))
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    open("results/kitsune_audit.txt", "w").write(buf.getvalue())
    print(buf.getvalue())
