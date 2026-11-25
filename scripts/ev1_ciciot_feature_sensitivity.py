"""CICIoT2023 sensitivity (post hoc, exploratory; nothing in results/ written): step-1 specialists
and IID with the registered selection minus some features (same partition, seed, trees).
For the DDoS_TCP_Flood specialist: the features it rests on and the share of each attack type
it flags. Variants: registered; minus IAT (F40); minus IAT and Header_Length (F2)."""
import os, sys
import numpy as np
os.chdir(os.path.expanduser("~/ereno-Adaptativo"))
sys.path.insert(0, "scripts")
import ev1_step1 as E
import xgboost as xgb

NAMES = ["flow_duration", "Header_Length", "Protocol Type", "Duration", "Rate", "Srate", "Drate",
         "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number",
         "ack_flag_number", "ece_flag_number", "cwr_flag_number", "ack_count", "syn_count",
         "fin_count", "urg_count", "rst_count", "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH",
         "IRC", "TCP", "UDP", "DHCP", "ARP", "ICMP", "IPv", "LLC", "Tot sum", "Min", "Max", "AVG",
         "Std", "Tot size", "IAT", "Number", "Magnitue", "Radius", "Covariance", "Variance", "Weight"]
dom, seed = E.DOMAINS["ciciot"], 42
feats, _ = E.selection(dom)
variants = [("registered", feats), ("minus IAT", [f for f in feats if f != 40]),
            ("minus IAT and Header_Length", [f for f in feats if f not in (40, 2)])]
rows = {}
for label, fs in variants:
    X, y, classes, nc = E.load(dom["train"], fs)
    units, slices, kept = E.attack_partition(y, nc, classes, dom["cap"], seed)
    S = len(units)
    spec = []
    for name, codes, node, idx in units:
        r = np.concatenate([idx, slices[node]])
        spec.append((name, node + 1, E.train(X[r], np.isin(y[r], codes).astype(np.int8), E.TREES, seed)[0]))
    iid = [E.train(X[i], (y[i] != nc).astype(np.int8), int(round(E.TREES * S / E.N)), seed)[0]
           for i in E.iid_partition(y, nc, dom["cap"], seed)]
    del X
    Xt, yt, _, _ = E.load(dom["test"], fs)
    d = xgb.DMatrix(Xt)
    del Xt
    H = np.column_stack([b.predict(d) >= 0.5 for _, _, b in spec])
    I = np.column_stack([b.predict(d) >= 0.5 for b in iid]).any(1)
    att, ben = yt != nc, yt == nc
    jt = [i for i, s in enumerate(spec) if s[0] == "DDoS_TCP_Flood"][0]
    g = spec[jt][2].get_score(importance_type="total_gain")
    tot = sum(g.values())
    top = sorted(g.items(), key=lambda kv: -kv[1])[:3]
    print(f"\n== {label}: {len(fs)} features")
    print(f"  attack-based OR  recall {100 * H.any(1)[att].mean():6.2f}  FPR {100 * H.any(1)[ben].mean():7.3f} %")
    print(f"  IID OR           recall {100 * I[att].mean():6.2f}  FPR {100 * I[ben].mean():7.3f} %")
    print(f"  DDoS_TCP_Flood specialist: FPR {100 * H[ben, jt].mean():.3f} %; top features "
          + ", ".join(f"{NAMES[fs[int(k[1:])] - 1]} {100 * v / tot:.0f} %" for k, v in top))
    rows[label] = {classes[c]: 100 * H[yt == c, jt].mean() for c in sorted(set(yt.tolist()) - {nc})}
    del H, I, d
print("\nshare of each attack type flagged by the DDoS_TCP_Flood specialist (%)")
print(f"{'type':<26}" + "".join(f"{l[:22]:>24}" for l, _ in variants))
for t in rows["registered"]:
    print(f"{t:<26}" + "".join(f"{rows[l][t]:>24.1f}" for l, _ in variants))
