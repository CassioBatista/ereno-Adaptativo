"""Audit of the GRASP incumbent defect (do_vnd overwrites the global best without comparing).

For every completed evaluation-v1 run (per-type and penalized global), the log is replayed
twice from its own lines:
  * buggy  - the code as run (python/grasp/vnd.py + local_searches.do_vnd); must reproduce the
             selection, F1 and iteration count stored in the run's JSON (parser validation);
  * fixed  - the global best is replaced only by a better solution, and an iteration counts as
             an improvement only if its VND result beats the global best.
The evaluation sequence does not depend on the global best (reconstruction starts from the
iteration-0 solution and the local searches never read it), so the fixed run is the same
run up to the point where either version stops.
"""
import glob, json, os, re, sys

ROOT = os.path.expanduser("~/ereno-Adaptativo")
LOGS = ["results/grasp_ev1.log", "results/grasp_ev1_morris4.log",
        "results/grasp_ev1_ciciot.log", "results/grasp_ev1_cicids.log"]
LAM = 0.05

re_ds_atk = re.compile(r"^\[grasp-atk\] dataset=all_in_one_ev1_([a-z0-9]+)_train")
re_ds_glob = re.compile(r"^\[global-pen\] dataset=all_in_one_ev1_([a-z0-9]+)_train")
re_atk = re.compile(r"^\[grasp-atk\] ATAQUE (\S+) \(\d+\).*parada (\d+) sem melhora")
re_atk_end = re.compile(r"^\[grasp-atk\] (\S+): (\d+) feats (\[.*?\]) f1_cv=([0-9.eE+-]+)")
re_glob_par = re.compile(r"^\[GRASP\].*parada: (\d+) itera")
re_glob_end = re.compile(r"^\[GRASP\] features \((\d+)\): (\[.*?\])\s+f1_cv=([0-9.eE+-]+)")
re_iter = re.compile(r"^######### ITERATION \((\d+)\)")
re_ev = re.compile(r"^EV;(\d+);([^;]*);(\[.*\])")
re_nova = re.compile(r"^\*\*\*\* nova:\s*(\[.*?\])\s*\(([^)]*)\)")


def fl(x):
    try:
        return float(x)
    except ValueError:
        return 0.0


class Sol:
    __slots__ = ("f", "f1")

    def __init__(self, f, f1):
        self.f, self.f1 = tuple(f), f1

    def n(self):
        return len(self.f)

    def pen(self):
        return self.f1 - LAM * self.n()

    def better(self, o):  # solution.py is_better_than, F1SCORE, penalized
        if o is None or o.n() == 0:
            return True
        if self.n() == 0:
            return False
        s, t = self.pen(), o.pen()
        if s == t:
            return self.n() < o.n()
        return s > t

    def key(self):
        return tuple(sorted(self.f))


def parse():
    secs = []
    for lp in LOGS:
        cur, ds = None, None
        for line in open(os.path.join(ROOT, lp), errors="replace"):
            if line.startswith("XGBoost;"):
                continue
            m = re_ds_atk.match(line)
            if m:
                ds = m.group(1); continue
            m = re_ds_glob.match(line)
            if m:
                ds = m.group(1)
                if cur: secs.append(cur)
                cur = dict(log=lp, ds=ds, name="GLOBAL", pat=None, iters=[], end=None)
                continue
            m = re_atk.match(line)
            if m:
                if cur: secs.append(cur)
                cur = dict(log=lp, ds=ds, name=m.group(1), pat=int(m.group(2)), iters=[], end=None)
                continue
            if cur is None:
                continue
            m = re_glob_par.match(line)
            if m and cur["name"] == "GLOBAL":
                cur["pat"] = int(m.group(1)); continue
            m = re_iter.match(line)
            if m:
                cur["iters"].append(dict(it=int(m.group(1)), seed=None, novas=[], nev=0)); continue
            m = re_ev.match(line)
            if m and cur["iters"]:
                it = cur["iters"][-1]
                it["nev"] += 1
                if it["seed"] is None:
                    it["seed"] = Sol(json.loads(m.group(3)), fl(m.group(2)))
                continue
            m = re_nova.match(line)
            if m and cur["iters"]:
                cur["iters"][-1]["novas"].append(Sol(json.loads(m.group(1)), fl(m.group(2))))
                continue
            m = re_atk_end.match(line)
            if m and cur["name"] == m.group(1):
                cur["end"] = (sorted(json.loads(m.group(3))), fl(m.group(4)))
                secs.append(cur); cur = None; continue
            m = re_glob_end.match(line)
            if m and cur["name"] == "GLOBAL":
                cur["end"] = (sorted(json.loads(m.group(2))), fl(m.group(3)))
                secs.append(cur); cur = None; continue
        if cur: secs.append(cur)
    return secs


def simulate(sec, fixed):
    """Returns (final Sol, number of loop iterations run, counter, stop index or None)."""
    its = sec["iters"]
    pat = sec["pat"]
    G, cnt, hist = None, 0, []
    for idx, it in enumerate(its):
        seed = it["seed"]
        best_local = seed
        if idx == 0:
            G = seed
        for nova in it["novas"]:
            if nova.better(best_local):
                best_local = nova
                if not fixed:
                    G = nova
        if idx == 0:
            if best_local.better(G):
                G = best_local
        else:
            if best_local.better(G):
                G = best_local; cnt = 0
            else:
                cnt += 1
        hist.append((it["it"], G, cnt))
        if idx > 0 and cnt >= pat:
            return G, idx, cnt, idx, hist
    return G, len(its) - 1, cnt, None, hist


def json_for(ds, name):
    if name == "GLOBAL":
        return os.path.join(ROOT, f"features/global_penalizado_ev1_{ds}_L0.05.json")
    return os.path.join(ROOT, f"features/por_ataque/grasppen005_ev1_{ds}_{name}.json")


def main():
    secs = [s for s in parse() if s["end"] is not None and s["iters"] and s["iters"][0]["seed"]]
    files = [f for f in glob.glob(os.path.join(ROOT, "features/por_ataque/grasppen005_ev1_*.json"))
             if "SUPERSET" not in f]
    files += glob.glob(os.path.join(ROOT, "features/global_penalizado_ev1_*_L0.05.json"))
    rows, problems = [], []
    for jf in sorted(files):
        d = json.load(open(jf))
        ds = re.search(r"ev1_([a-z0-9]+)", d["dataset"]).group(1)
        name = "GLOBAL" if "global_penalizado" in jf else d["attack"]
        jsel = sorted(d["features"])
        jf1 = d.get("f1_grasp_cv", d.get("f1_cv"))
        cand = [s for s in secs if s["ds"] == ds and s["name"] == name and s["end"][0] == jsel
                and abs(s["end"][1] - jf1) < 1e-9]
        if not cand:
            problems.append((ds, name, "no completed log section matches the JSON")); continue
        s = cand[-1]
        if s["pat"] is None:
            s["pat"] = d.get("no_improvement", 5)
        Gb, nb, cb, stop_b, _ = simulate(s, fixed=False)
        ok = Gb.key() == tuple(jsel) and abs(Gb.f1 - jf1) < 1e-9 and stop_b is not None
        if not ok:
            problems.append((ds, name, f"buggy replay does not reproduce the JSON: sim {sorted(Gb.f)} "
                             f"{Gb.f1} stop={stop_b} cnt={cb}/{s['pat']} vs json {jsel} {jf1}"))
        Gf, nf, cf, stop_f, histf = simulate(s, fixed=True)
        # best evaluated subset among the iteration results actually run (upper reference)
        if stop_f is None:
            status = "EXTEND"
        elif Gf.key() == Gb.key():
            status = "same"
        else:
            status = "DIFF"
        rows.append(dict(ds=ds, name=name, pat=s["pat"], loop_b=nb, loop_f=nf, cnt_f=cf,
                         stop_f=stop_f, b=Gb, f=Gf, status=status, log=s["log"], valid=ok))
    order = {"morris4": 0, "ciciot": 1, "cicids": 2, "morris3": 3}
    rows.sort(key=lambda r: (order.get(r["ds"], 9), r["name"] != "GLOBAL", r["name"]))
    print(f"{len(rows)} runs audited; {sum(r['valid'] for r in rows)} buggy replays reproduce the JSON")
    for p in problems:
        print("PROBLEM:", p)
    from collections import Counter
    print("status:", dict(Counter((r["ds"], r["status"]) for r in rows)))
    print()
    hdr = f"{'domain':8} {'run':28} {'pat':>3} {'it_b':>4} {'it_f':>4} {'st':6}  as run (n, F1, pen)            fixed (n, F1, pen)"
    print(hdr)
    for r in rows:
        b, f = r["b"], r["f"]
        print(f"{r['ds']:8} {r['name'][:28]:28} {r['pat']:3d} {r['loop_b']:4d} "
              f"{(r['loop_f'] if r['stop_f'] is not None else str(r['loop_f'])+'+'):>4} {r['status']:6}  "
              f"{b.n():2d} {b.f1:9.4f} {b.pen():9.4f}   {f.n():2d} {f.f1:9.4f} {f.pen():9.4f}"
              + ("" if r["valid"] else "  [replay mismatch]"))
    print()
    for r in rows:
        if r["status"] != "same":
            print(f"{r['ds']} {r['name']}: as run {sorted(r['b'].f)} -> fixed {sorted(r['f'].f)}"
                  + (f" (fixed run would continue; counter {r['cnt_f']}/{r['pat']} at the last logged iteration)"
                     if r["status"] == "EXTEND" else ""))


if __name__ == "__main__":
    main()
