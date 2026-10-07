#!/usr/bin/env python3
"""Window triage keyed by attribution, on the corrected CICIDS2017.

The ERENO policy counts the flagged flows of a window over the whole network (volume) and
their share (fraction); on CICIDS2017-C it catches 26.9 % of the attack windows because
slow attacks flag about one flow per second, the level of benign false alarms. Here the
count is taken per attribution key (the profile's emitter, Src IP for this dataset): a
slow attacker accumulates its own flags, whereas benign false alarms spread over hosts.
With a single key (ERENO: one publisher identity) or no attribution, the per-source count
equals the global one, so the rule generalizes the ERENO policy.

Caveat (user, 2026-10-07): in this capture every external attack enters through the same
gateway (172.16.0.1, NAT), so a per-source count may learn the capture topology rather
than the attacker. The script therefore also tests the key the architecture itself
provides, the SPECIALIST: per window, the fused-flagged flows (k >= 2) on which a
specialist of each category voted, with a per-category threshold. It needs no address,
exists under every profile (attribution none included) and names the attack category.
It reports the gateway's share of attack and benign flows as a diagnostic.

Same split and specialists as cross_domain_results.py (k >= 2). Thresholds use the ERENO
margin policy on Monday's first 70 % (benign, never used for training):
  T_src = ceil(1.25 * max per-source flags in a calibration window) + 1.
Reports, for 1, 10 and 60 s windows: attack-window recall, benign-window false-alarm rate,
per category, and whether the alarming source is an attack source of that window.

  python scripts/_cicids_source_triage.py
Out: results/cicids2017c_source_triage.txt
"""
import io
import math
import os
import sys
import zipfile
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cross_domain_results as C  # noqa: E402

WLENS = (1, 10, 60)
FLOOR = 2          # minimum per-specialist threshold (ERENO check: scripts/oos_specialist_triage.py)


def load():
    # one day at a time, keeping only the feature matrix and a few light columns (the full
    # text frame of five days peaked at 6.8 GB in a 9 GB VM shared with the GRASP run)
    z = zipfile.ZipFile(C.D + "cicids2017_improved/CICIDS2017_improved.zip")
    Xs, lights = [], []
    for d in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
        dd = pd.read_csv(z.open(f"{d}.csv"), low_memory=False, dtype={c: "str" for c in C.TEXT})
        dd["_cat"] = dd["Label"].map(C.cat_cicids)
        dd = dd[dd["_cat"] != "drop"].reset_index(drop=True)
        feats = [c for c in dd.columns if c not in C.TEXT + C.IDENT + C.FINGERPRINT + ["_cat"]]
        Xd = dd[feats].apply(pd.to_numeric, errors="coerce").astype("float32").to_numpy()
        Xd[~np.isfinite(Xd)] = np.nan
        Xs.append(Xd)
        lights.append(pd.DataFrame({"_day": d, "_cat": dd["_cat"].values, "Src IP": dd["Src IP"].values,
                                    "_ts": pd.to_datetime(dd["Timestamp"], format="mixed").values}))
        del dd
    X = np.concatenate(Xs); del Xs
    df = pd.concat(lights, ignore_index=True); del lights
    test = np.zeros(len(df), bool)
    for d, g in df.groupby("_day"):
        if d == "monday":
            continue
        for c, gc in g.groupby("_cat"):
            if c != "Benign":
                t70, tmax = gc["_ts"].quantile(0.7), gc["_ts"].max()
                test |= ((df["_day"] == d) & (df["_ts"] >= t70) & (df["_ts"] <= tmax)).to_numpy()
    mon = (df["_day"] == "monday").to_numpy()
    t70m = df.loc[mon, "_ts"].quantile(0.7)
    calib = mon & (df["_ts"] < t70m).to_numpy()
    test |= mon & (df["_ts"] >= t70m).to_numpy()
    train = ~test & ~calib
    secs = (df["_ts"] - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy()
    src = pd.factorize(df["Src IP"])[0]
    return X, df["_cat"].to_numpy(), secs, src, df["Src IP"].to_numpy(), train, calib, test


def per_window(secs, flag, src, wlen):
    """Per window: total flags and the largest per-source flag count (and that source)."""
    w = np.floor(secs / wlen).astype(np.int64)
    tot = pd.Series(flag).groupby(w).sum()
    f = flag.astype(bool)
    if f.any():
        cnt = pd.DataFrame({"w": w[f], "s": src[f]}).value_counts()
        cnt = cnt.reset_index().sort_values("count", ascending=False).drop_duplicates("w")
        best = cnt.set_index("w")
    else:
        best = pd.DataFrame(columns=["s", "count"])
    out = pd.DataFrame({"tot": tot})
    out["n"] = pd.Series(1, index=w).groupby(level=0).sum()
    out["src_max"] = best["count"].reindex(out.index).fillna(0).astype(int)
    out["src_arg"] = best["s"].reindex(out.index).fillna(-1).astype(int)
    return w, out


def specialist_counts(secs, V, fused, wlen, ncat):
    """Per window and category: fused-flagged flows (k >= 2) on which a specialist of
    that category voted. Key = the specialist, available in every profile."""
    w = np.floor(secs / wlen).astype(np.int64)
    cv = np.column_stack([V[:, [c, c + ncat]].max(1) for c in range(ncat)]) * fused[:, None]
    return pd.DataFrame(cv, columns=range(ncat)).groupby(w).sum()


def main():
    X, cat, secs, src, srcip, train, calib, test = load()
    boosters, _ = C.train_specialists(X[train], cat[train], C.CATS_C)
    ncat = len(C.CATS_C)                 # node i serves category i % 7 (pair i // 7)
    Vc, Vt = C.votes(boosters, X[calib]), C.votes(boosters, X[test])
    fc = (Vc.sum(1) >= 2).astype(np.int8)
    ft = (Vt.sum(1) >= 2).astype(np.int8)
    cte, ste, tte = cat[test], src[test], secs[test]
    gw = srcip == "172.16.0.1"
    print(f"gateway 172.16.0.1 as source: {100 * gw[test & (cat != 'Benign')].mean():.1f}% of test "
          f"attack flows, {100 * gw[test & (cat == 'Benign')].mean():.2f}% of test benign flows, "
          f"{100 * gw[calib].mean():.2f}% of calibration flows")
    print(f"calibration flows {int(calib.sum()):,} (flagged {int(fc.sum()):,}); "
          f"test flows {int(test.sum()):,} (flagged {int(ft.sum()):,})")
    for wlen in WLENS:
        _, wc = per_window(secs[calib], fc, src[calib], wlen)
        T_glob = int(math.ceil(1.25 * wc["tot"].max())) + 1
        f_glob = 1.25 * (wc["tot"] / wc["n"]).max()
        T_src = int(math.ceil(1.25 * wc["src_max"].max())) + 1
        w, wt = per_window(tte, ft, ste, wlen)
        # window label and its attack sources
        lab = pd.DataFrame({"w": w, "c": cte, "s": ste})
        atk_rows = lab[lab["c"] != "Benign"]
        wcat = atk_rows.groupby("w")["c"].agg(lambda s: s.mode().iloc[0])
        wt["cat"] = wcat.reindex(wt.index).fillna("Benign")
        asrc = atk_rows.groupby("w")["s"].agg(set)
        atk = (wt["cat"] != "Benign").values
        a_glob = (wt["tot"] >= T_glob) | ((wt["tot"] / wt["n"]) >= f_glob) if f_glob < 1 else wt["tot"] >= T_glob
        a_src = wt["src_max"] >= T_src
        # per specialist: per-category thresholds from the benign calibration
        sc = specialist_counts(secs[calib], Vc, fc, wlen, ncat)
        st = specialist_counts(tte, Vt, ft, wlen, ncat).reindex(wt.index).fillna(0)
        floor_rows = []
        for floor in (1, 2, 3):        # a category never flagged in calibration gets T = floor
            Tf = np.maximum(np.ceil(1.25 * sc.max().values) + 1, floor)
            hf = (st.values >= Tf).any(1)
            floor_rows.append((floor, hf))
        T_cat = np.maximum(np.ceil(1.25 * sc.max().values) + 1, FLOOR)
        hit_cat = st.values >= T_cat
        a_spec = pd.Series(hit_cat.any(1), index=wt.index)
        # alarm names the category that exceeds its own threshold the most
        ratio = np.where(hit_cat, st.values / T_cat, -1)
        wt["spec_cat"] = np.where(hit_cat.any(1), np.array(C.CATS_C)[ratio.argmax(1)], "-")
        print(f"\n=== window {wlen} s: test {int(atk.sum()):,} attack / {int((~atk).sum()):,} benign windows")
        print(f"  calibration: max global flags {wc['tot'].max()}, max per-source flags "
              f"{wc['src_max'].max()} -> T_glob={T_glob}, f_glob={f_glob:.2f}, T_src={T_src}")
        print("  per-specialist thresholds: "
              + ", ".join(f"{c}={int(t)}" for c, t in zip(C.CATS_C, T_cat)))
        for floor, hf in floor_rows:
            g = a_glob.values | hf
            print(f"  floor {floor}: per specialist {100 * hf[atk].mean():5.1f}% / FA {int(hf[~atk].sum())};"
                  f"  global OR per specialist {100 * g[atk].mean():5.1f}% / FA {int(g[~atk].sum())} "
                  f"({100 * g[~atk].mean():.3f}%)")
        print(f"  (rows below: per-specialist floor {FLOOR})")
        for name, a in (("global (ERENO policy)", a_glob), ("per source", a_src),
                        ("per specialist", a_spec), ("global OR per specialist", a_glob | a_spec),
                        ("global OR per source", a_glob | a_src)):
            print(f"  {name:<24} attack windows {100 * a[atk].mean():5.1f}%   "
                  f"benign false alarms {int(a[~atk].sum()):>4} ({100 * a[~atk].mean():.3f}%)")
        print(f"  threshold curve, per source:  T  attack-win  benign-FA")
        for T in sorted({2, 3, 5, 10, 20, T_src}):
            print(f"  {'':<30}{T:>3}  {100 * (wt['src_max'][atk] >= T).mean():8.1f}%  "
                  f"{100 * (wt['src_max'][~atk] >= T).mean():8.3f}%")
        print("  recall per category of the window: global | per source | per specialist "
              "(and alarm names the right category)")
        for c in C.CATS_C:
            m = (wt["cat"] == c).values
            if m.any():
                right = (wt["spec_cat"].values[m] == c).mean()
                print(f"    {c:<13} windows={int(m.sum()):>5}  {100 * a_glob[m].mean():5.1f}% | "
                      f"{100 * a_src[m].mean():5.1f}% | {100 * a_spec[m].mean():5.1f}% "
                      f"(right category {100 * right:5.1f}%)")
        hit = wt.index[a_src.values & atk]
        ok = sum(wt.at[i, "src_arg"] in asrc.get(i, set()) for i in hit)
        gwid = src[np.argmax(gw)] if gw.any() else -2
        via_gw = int((wt.loc[hit, "src_arg"] == gwid).sum())
        print(f"  per source, attribution: alarming source is an attack source in {ok}/{len(hit)} "
              f"detected attack windows; the gateway 172.16.0.1 in {via_gw}/{len(hit)}")
        ipof = pd.Series(srcip, index=src).groupby(level=0).first()
        fa = wt.loc[a_src.values & ~atk, "src_arg"]
        print(f"  per source, benign false-alarm windows by alarming source: "
              f"{fa.map(ipof).value_counts().head(3).to_dict()}")


if __name__ == "__main__":
    os.chdir(os.path.expanduser("~/ereno-Adaptativo"))
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    open("results/cicids2017c_source_triage.txt", "w").write(buf.getvalue())
    print(buf.getvalue())
