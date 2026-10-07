#!/usr/bin/env python3
"""Do the low-volume CICIoT2023 classes contain the attack, or ordinary device traffic
recorded during the attack session (as found in CIC-BCCC-NRC-IoMT-2024)?

CICIoT2023 rows carry protocol indicators (HTTP, HTTPS, DNS, Telnet, SMTP, SSH, IRC, TCP,
UDP, DHCP, ARP, ICMP, IPv, LLC; averages over the row's packet window). For each label the
script reports the protocol mix and, for attacks with an expected protocol, the share of
rows that show it at all. It then checks where the benign false positives of the
specialists sit: which benign rows are flagged, and how they compare to the attack rows.
Out: results/ciciot2023_label_protocol_check.txt
"""
import glob
import io
import os
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inspect_ciciot2023 import category  # noqa: E402

D = os.path.expanduser("~/datasets/ciciot2023/")
PROTO = ["HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP", "ARP",
         "ICMP", "IPv", "LLC"]
EXPECT = {"MITM-ArpSpoofing": ["ARP"], "DNS_Spoofing": ["DNS"], "Recon-PingSweep": ["ICMP"],
          "Recon-HostDiscovery": ["ARP", "ICMP"], "SqlInjection": ["HTTP", "HTTPS"],
          "CommandInjection": ["HTTP", "HTTPS"], "XSS": ["HTTP", "HTTPS"],
          "Uploading_Attack": ["HTTP", "HTTPS"], "Backdoor_Malware": ["HTTP", "HTTPS"],
          "BrowserHijacking": ["HTTP", "HTTPS"], "DictionaryBruteForce": ["HTTP", "HTTPS", "Telnet", "SSH"],
          "DDoS-ICMP_Flood": ["ICMP"], "DDoS-UDP_Flood": ["UDP"], "DoS-UDP_Flood": ["UDP"],
          "DDoS-HTTP_Flood": ["HTTP", "HTTPS"], "DoS-HTTP_Flood": ["HTTP", "HTTPS"]}


def main():
    df = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(D + "part-*.csv"))], ignore_index=True)
    lab = df.columns[-1]
    df["_cat"] = df[lab].map(category)
    print(f"rows={len(df):,}")
    print("\n=== protocol mix per label (mean indicator; 'any' = share of rows with indicator > 0)")
    g = df.groupby(lab)
    mix = g[PROTO].mean()
    order = df[lab].value_counts().index
    print(mix.loc[order].round(3).to_string())

    print("\n=== attacks with an expected protocol: share of rows that show it")
    for l, protos in EXPECT.items():
        s = df[df[lab] == l]
        if len(s):
            has = (s[protos] > 0).any(axis=1).mean()
            print(f"  {l:<22} n={len(s):>8,}  rows with {'/'.join(protos):<22} {100 * has:6.1f}%")
    b = df[df[lab] == "BenignTraffic"]
    print(f"  {'BenignTraffic':<22} n={len(b):>8,}  rows with HTTP/HTTPS {100 * (b[['HTTP', 'HTTPS']] > 0).any(axis=1).mean():6.1f}%  "
          f"ARP {100 * (b['ARP'] > 0).mean():.1f}%  DNS {100 * (b['DNS'] > 0).mean():.1f}%  "
          f"ICMP {100 * (b['ICMP'] > 0).mean():.1f}%")

    print("\n=== nearest benign look-alikes: share of each low-volume class whose rows are "
          "closer to a benign row than to any other row of their own class (sample, standardized)")
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    feats = [c for c in df.columns if c not in (lab, "_cat", "IAT")]
    rng = np.random.default_rng(0)
    ben = df[df["_cat"] == "Benign"].sample(min(20000, int((df["_cat"] == "Benign").sum())), random_state=0)
    sc = StandardScaler().fit(df[feats].sample(200000, random_state=0).to_numpy())
    Xb = sc.transform(ben[feats].to_numpy())
    nnb = NearestNeighbors(n_neighbors=1).fit(Xb)
    for c in ["Recon", "Web", "BruteForce", "Spoofing", "DDoS", "DoS", "Mirai"]:
        s = df[df["_cat"] == c]
        s = s.sample(min(5000, len(s)), random_state=0)
        Xs = sc.transform(s[feats].to_numpy())
        db, _ = nnb.kneighbors(Xs)
        own = NearestNeighbors(n_neighbors=2).fit(Xs)
        do, _ = own.kneighbors(Xs)
        closer = (db[:, 0] < do[:, 1]).mean()
        print(f"  {c:<11} n={len(s):>5}  nearer to benign than to own class: {100 * closer:5.1f}%  "
              f"median dist to benign {np.median(db):.3f} vs own {np.median(do[:, 1]):.3f}")


if __name__ == "__main__":
    os.chdir(os.path.expanduser("~/ereno-Adaptativo"))
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    open("results/ciciot2023_label_protocol_check.txt", "w").write(buf.getvalue())
    print(buf.getvalue())
