#!/usr/bin/env python3
"""Fan a federation-level event stream out into 15 PER-INSTANCE streams.

One monitored instance per ReSIDS agent: the FL server plus the 14 clients, each with its
own endpoint and its own independent `seq` space (docs/MULTI_INSTANCE.md). The monitor then
holds 15 watermarks instead of one, which is the whole difference.

Two observation views are produced from the same scored trace, because they answer
different questions and only one of them is what we measured:

  replicated  every client scores the WHOLE window with the same diffused booster union,
              so all live clients reach the SAME verdict. This is faithful to the
              simulation: one ERENO test split, scored by everyone. Duplicate alarms are
              therefore AGREEMENT, not independent corroboration, and the monitor's value
              is detecting DIVERGENCE (an instance that disagrees, or has gone stale).

  sharded     the window's samples are partitioned round-robin across the 14 clients, so
              each agent observes only its own slice -- the deployment case, where an IED
              agent sees its own bus segment. Alarms from different instances are then
              INDEPENDENT evidence about the same window, and k-of-n corroboration AT THE
              MONITOR becomes meaningful. The booster union is still complete everywhere;
              only the observation is split.

The control plane (node_failure, node_recovery, architecture_change, node_isolated) is
identical in both views: it is not a detection result.

Asymmetries of the server instance, which a naive 15-instance monitor gets wrong:
  * it scores no traffic, so it emits NO alarms -- it is a control-plane target only;
  * in GL it has no role and goes SILENT. That silence is expected, not a fault. A monitor
    that treats 15 endpoints uniformly raises a false alarm on the server immediately
    after the very FL->GL switch it just observed;
  * an ISOLATED client keeps emitting. Isolation removes it from the federation, not from
    the bus, so its agent runs on with a frozen view of membership -- which is exactly the
    divergence the monitor should see, and the confirmation that isolation took effect at
    the federation level while the node stayed live.

Reads the committed federation stream for the control-plane timeline (so the two can never
disagree) and recomputes only the detection part per instance.

  python scripts/instance_streams.py conf/scenarios/intrusion.yaml
Out: results/instances/<scenario>_<view>/{srv,c00..c13}.jsonl
     results/instance_corroboration_<scenario>.csv
"""
import argparse
import csv
import datetime as dt
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import xgboost as xgb

import python.util as util
from provenance import file_ref, write_manifest
from scenario_events import SIM_EPOCH, iso, load_scenario

CONTROL = {"node_failure", "node_recovery", "architecture_change", "node_isolated"}


def train_and_score(sc):
    """Mirrors the training block of scenario_events.py deliberately: that script's outputs
    are committed and hash-pinned, so it is left untouched here."""
    B, M, N = sc["data"], sc["model"], sc["nodes"]
    feats = list(M["features"])
    params = {"max_depth": M["max_depth"], "eta": M["eta"], "objective": "binary:logistic",
              "seed": M["seed"], "nthread": 4}
    rng = np.random.default_rng(M["seed"])
    nc = util.normal_class
    print(f"[inst] loading train ({B['train']}, {len(feats)} features)...")
    Xtr_raw, ytr, cv = util.load_arff(f"{B['train']}.csv")
    Xtr = util.filter_features(Xtr_raw, feats); del Xtr_raw
    nidx = np.where(ytr == nc)[0]
    if len(nidx) > B["benign_cap"]:
        drop = rng.permutation(nidx)[B["benign_cap"]:]
        keep = np.ones(len(ytr), bool); keep[drop] = False
        Xtr, ytr = Xtr[keep], ytr[keep]
    attacks = sorted(int(c) for c in np.unique(ytr) if c != nc)
    bslice = np.array_split(rng.permutation(np.where(ytr == nc)[0]), N)
    spec_attack = []
    for at in attacks:
        spec_attack += [at, at]
    spec_attack = spec_attack[:N]

    print(f"[inst] training {N} specialists...")
    boosters = []
    for cid in range(N):
        at = spec_attack[cid]
        aidx = np.where(ytr == at)[0]
        peers = [c for c in range(N) if spec_attack[c] == at]
        shard = np.array_split(rng.permutation(aidx), len(peers))[peers.index(cid)]
        idx = np.concatenate([shard, bslice[cid]])
        lab = (ytr[idx] != nc).astype(int)
        p = dict(params, scale_pos_weight=(lab == 0).sum() / max(lab.sum(), 1))
        boosters.append(xgb.train(p, xgb.DMatrix(Xtr[idx], label=lab),
                                  num_boost_round=M["num_boost_round"]))
    del Xtr, ytr

    print("[inst] scoring test...")
    Xte_raw, yte, _ = util.load_arff(f"{B['test']}.csv")
    t_te = Xte_raw[:, 0].copy()
    Xte = util.filter_features(Xte_raw, feats); del Xte_raw
    d = xgb.DMatrix(Xte); del Xte
    fired = np.vstack([(b.predict(d) >= 0.5) for b in boosters])
    order = np.argsort(t_te, kind="stable")
    return t_te[order], fired[:, order], yte[order], nc, cv, spec_attack


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("--views", default="replicated,sharded")
    a = ap.parse_args()
    sc = load_scenario(a.scenario)
    name, N, K = sc["name"], sc["nodes"], sc["fusion_k"]
    fed_path = f"results/events_{name}.jsonl"
    fed = [json.loads(l) for l in open(fed_path, encoding="utf-8")]
    print(f"[inst] {name}: {len(fed)} federation events from {fed_path}")

    ts, fired, y, nc, cv, spec_attack = train_and_score(sc)
    t0 = float(ts[0])
    win = np.floor((ts - t0) / sc["window_seconds"]).astype(np.int64)
    bnd = np.searchsorted(win, np.arange(int(win[-1]) + 2))

    views = a.views.split(",")
    # one independent seq space per instance, per view
    inst = ["srv"] + [f"c{c:02d}" for c in range(N)]
    out = {v: {i: [] for i in inst} for v in views}
    seqs = {v: {i: 0 for i in inst} for v in views}
    # an isolated client keeps running with the membership view it had when it was cut off
    frozen = {}
    isolated, failed = set(), set()
    corr_rows = []

    def emit(view, who, ev):
        seqs[view][who] += 1
        out[view][who].append(dict(ev, seq=seqs[view][who], instance=who))

    for ev in fed:
        t = ev["type"]
        mode = ev.get("mode")
        active = set(ev.get("active_nodes") or range(N))

        if t == "node_failure":
            failed |= set(ev.get("failed_nodes", []))
        if t == "node_recovery":
            failed -= set(ev.get("recovered_nodes", []))
        if t == "node_isolated" and ev.get("source_node") is not None:
            isolated.add(ev["source_node"])
            frozen[ev["source_node"]] = {"mode": mode, "n_active": ev.get("n_active")}

        live = [c for c in range(N) if c not in failed]

        if t in CONTROL:
            for v in views:
                # the server records control-plane facts while it has a role; after FL->GL
                # it is out, and its silence from here on is correct behaviour
                if mode != "gossip" or t == "architecture_change":
                    emit(v, "srv", dict(ev, instance_role="aggregator"))
                for c in live:
                    e = dict(ev)
                    if c in isolated and c in frozen:
                        e["mode"] = frozen[c]["mode"]
                        e["n_active"] = frozen[c]["n_active"]
                        e["detail"] = ("emitted by an ISOLATED instance: its membership view "
                                       "is frozen at isolation, which is how the monitor "
                                       "confirms the isolation took effect while the node "
                                       "stayed live")
                    emit(v, f"c{c:02d}", e)
            continue

        if t != "intrusion_detected":
            continue

        # --- detection: recompute per instance -------------------------------------
        secs = (dt.datetime.fromisoformat(ev["window_start"]) - SIM_EPOCH).total_seconds()
        w = int(round((secs - t0) / sc["window_seconds"]))
        lo, hi = int(bnd[w]), int(bnd[w + 1])
        pool = sorted(active) if mode == "federated" else sorted(range(N))
        sub = fired[pool, lo:hi]
        votes = sub.sum(axis=0)
        fl = votes >= K
        nw = hi - lo

        fired_by = {}
        for c in live:
            if "replicated" in views:
                fired_by.setdefault("replicated", {})[c] = (int(fl.sum()), int(votes[fl].max()) if fl.any() else 0)
            if "sharded" in views:
                m = fl[c::N]
                vv = votes[c::N][m]
                fired_by.setdefault("sharded", {})[c] = (int(m.sum()), int(vv.max()) if m.any() else 0)

        for v in views:
            n_fire = 0
            for c in live:
                nflags, kv = fired_by[v][c]
                if nflags == 0:
                    continue
                n_fire += 1
                e = dict(ev, n_flags=nflags, k_votes=kv, window_samples=nw)
                if c in isolated and c in frozen:
                    e["mode"] = frozen[c]["mode"]
                    e["n_active"] = frozen[c]["n_active"]
                emit(v, f"c{c:02d}", e)
            corr_rows.append({"view": v, "window": w, "round": ev["round"],
                              "fed_seq": ev["seq"], "attack": ev["attack"],
                              "benign_window": int(bool((y[lo:hi] == nc).all())),
                              "n_samples": nw, "live_clients": len(live),
                              "instances_firing": n_fire,
                              "fed_n_flags": ev["n_flags"], "fed_k_votes": ev["k_votes"]})

    os.makedirs("results/instances", exist_ok=True)
    for v in views:
        d = f"results/instances/{name}_{v}"
        os.makedirs(d, exist_ok=True)
        tot = 0
        for i in inst:
            with open(f"{d}/{i}.jsonl", "w", encoding="utf-8") as fh:
                for e in out[v][i]:
                    fh.write(json.dumps(e) + "\n")
            tot += len(out[v][i])
        nz = sum(1 for i in inst if out[v][i])
        print(f"[inst] {v:11s}: {tot:6d} events across {nz}/{len(inst)} instances -> {d}/")
        print(f"[inst]              srv={len(out[v]['srv'])}  "
              f"clients={[len(out[v][f'c{c:02d}']) for c in range(N)]}")

    cpath = f"results/instance_corroboration_{name}.csv"
    with open(cpath, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(corr_rows[0]))
        wr.writeheader(); wr.writerows(corr_rows)
    print(f"[inst] corroboration census -> {cpath}")

    # --- the comparison the two views exist for -----------------------------------
    print(f"\n[inst] how many of {N} instances fire on the same window")
    for v in views:
        rows = [r for r in corr_rows if r["view"] == v]
        for label, sel in (("attack windows", [r for r in rows if not r["benign_window"]]),
                           ("benign windows (every alarm is a FALSE POSITIVE)",
                            [r for r in rows if r["benign_window"]])):
            if not sel:
                continue
            dist = Counter(r["instances_firing"] for r in sel)
            print(f"  {v:11s} {label}: n={len(sel)}")
            print(f"              distribution {dict(sorted(dist.items()))}")
            for k in (1, 2, 3, 5):
                hit = sum(n for i, n in dist.items() if i >= k)
                print(f"              >={k} instances: {hit:4d} ({100*hit/len(sel):5.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
