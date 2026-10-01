#!/usr/bin/env python3
"""The external monitor, as a 15-instance consumer of the ReSIDS API.

Polls 15 endpoints over real HTTP — the FL server and the 14 clients — and does the four
things a single-endpoint monitor never has to:

1. FIFTEEN WATERMARKS. Each instance has its own independent `seq` space, so there is one
   contiguous watermark per instance and a `since` from one is meaningless on another. The
   gap rule of docs/API.md §2.1 is applied per instance, for the same measured reason.

2. THE SERVER'S SILENCE IS NOT A FAULT. In GL the aggregator has no role and stops
   emitting. A monitor that treats 15 endpoints uniformly declares it dead immediately
   after the very FL->GL switch it just observed. The rule here is derived from the events
   rather than special-cased: while the federation mode is gossip, srv is expected quiet.

3. AGREEMENT vs DIVERGENCE. Alarms for the same window arrive from many instances. Under
   the replicated view they are the same verdict reached independently, so the useful
   signal is not the count but a MISMATCH -- an instance reporting a different attack, or
   carrying a stale membership view (which is what an isolated-but-live node does). Under
   the sharded view the count IS evidence, and k-of-n across instances applies.

4. FAN-OUT WITH AN ASYMMETRIC UNANIMITY RULE. A mode switch is federation-wide but
   commands are per-instance, so a fan-out can partially fail. FL->GL may proceed
   partially: the stragglers converge on GL, which is the safe resting state. GL->FL may
   NOT -- a partial return splits the federation, some instances aggregating through a hub
   that others have abandoned. So the monitor pre-checks every instance and aborts the
   whole intent unless all 15 would accept. This falls straight out of the authority
   asymmetry (docs/decentralized_monitoring.md §1.1) and is the one place where the
   15-instance setting changes the command semantics rather than just repeating them.

  python scripts/multi_instance_monitor.py --seconds 30
Out: results/multi_instance_monitor_<tag>.txt
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict

N = 14
ESCALATE = 275          # n_flags triage threshold from the FP baseline (removes every FP)


def get(base, path, timeout=3.0):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return json.load(r)


class Instance:
    """One monitored ReSIDS agent: its endpoint, its watermark, its health."""

    def __init__(self, name, base):
        self.name, self.base = name, base
        self.seen, self.contiguous = set(), 0
        self.events, self.alive, self.last_ok = 0, None, None
        self.mode, self.round, self.n_active = None, None, None
        self.silent_polls = 0

    def poll(self):
        try:
            new = get(self.base, f"/events?since={self.contiguous}")
            st = get(self.base, "/status")
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            self.alive = False
            return [], str(e)
        self.alive, self.last_ok = True, time.time()
        self.mode, self.round = st.get("current_mode"), st.get("round")
        self.n_active = len(st.get("active_nodes") or [])
        for ev in new:
            self.seen.add(ev["seq"])
        while self.contiguous + 1 in self.seen:
            self.contiguous += 1
        self.events += len(new)
        self.silent_polls = self.silent_polls + 1 if not new else 0
        return new, None

    @property
    def gap(self):
        """An empty answer means alive, no gap and nothing to recover -- per instance."""
        return len(self.seen) != self.contiguous


def fan_out_plan(insts, to_mode):
    """What the monitor would send, and whether it may send it at all."""
    unanimity = (to_mode == "federated")
    reachable = [i for i in insts if i.alive]
    ok = len(reachable) == len(insts)
    plan = {
        "intent": f"set_mode -> {to_mode}",
        "targets": len(insts),
        "reachable": len(reachable),
        "unanimity_required": unanimity,
        "may_proceed": ok or not unanimity,
    }
    if unanimity and not ok:
        plan["abort_reason"] = (
            f"GL->FL needs all {len(insts)} instances to accept; {len(insts) - len(reachable)} "
            f"unreachable. A partial return splits the federation: some instances would "
            f"aggregate through a hub the others have abandoned. Staying in GL is the safe "
            f"option, and by design it has no deadline fallback.")
    elif not unanimity and not ok:
        plan["note"] = (
            f"FL->GL proceeds with {len(reachable)}/{len(insts)}: the unreachable instances "
            f"converge on GL anyway, which is the resting state. Partial application is "
            f"safe in this direction only.")
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--base-port", type=int, default=8722)
    ap.add_argument("--every", type=float, default=1.0)
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()

    names = ["srv"] + [f"c{c:02d}" for c in range(N)]
    insts = [Instance(nm, f"http://{a.host}:{a.base_port + k}") for k, nm in enumerate(names)]
    by_name = {i.name: i for i in insts}

    # per-window agreement, built as alarms arrive from the 15 endpoints
    window_reports = defaultdict(dict)       # window_start -> {instance: (attack, n_flags, k)}
    divergence, control, fed_mode = [], [], "federated"
    incidents = 0

    print(f"[monitor] polling {len(insts)} instances on "
          f"{a.host}:{a.base_port}-{a.base_port + len(insts) - 1}")
    print("    t   alive  events  srv      mode        windows  diverg  incidents")
    t0 = time.time()
    while time.time() - t0 < a.seconds:
        for i in insts:
            new, err = i.poll()
            for ev in new:
                if ev["type"] == "intrusion_detected":
                    window_reports[ev["window_start"]][i.name] = (
                        ev["attack"], ev.get("n_flags"), ev.get("k_votes"),
                        ev.get("mode"), ev.get("n_active"))
                    if (ev.get("n_flags") or 0) >= ESCALATE:
                        incidents += 1
                else:
                    control.append((i.name, ev["round"], ev["type"], ev.get("decided_by")))
                    if ev["type"] == "architecture_change":
                        fed_mode = ev["to_mode"]

        # the server is expected quiet in GL; only flag it when it owes us a round
        srv = by_name["srv"]
        srv_state = ("quiet (expected in GL)" if fed_mode == "gossip"
                     else "down" if srv.alive is False
                     else "up" if srv.events or srv.silent_polls < 3 else "SILENT")
        alive = sum(1 for i in insts if i.alive)
        evs = sum(i.events for i in insts)
        print(f"{time.time()-t0:5.1f}s  {alive:2d}/{len(insts)}  {evs:6d}  "
              f"{srv_state:22s}  {fed_mode:10s}  {len(window_reports):7d}  "
              f"{len(divergence):6d}  {incidents:9d}")
        time.sleep(a.every)

    # ---- agreement / divergence census ------------------------------------------
    # Divergence is only useful if it names the MINORITY. Flagging a window and then
    # listing all 14 instances says nothing: the question is which of them disagrees with
    # the rest, because that one is reporting something real (a stale membership view is
    # the signature of a node isolated from the federation but still running).
    agree_hist, stale_views, label_split = Counter(), Counter(), Counter()
    minority_view = {}
    for wstart, reps in window_reports.items():
        clients = {k: v for k, v in reps.items() if k != "srv"}
        agree_hist[len(clients)] += 1
        if len(clients) < 2:
            continue
        labels = Counter(v[0] for v in clients.values())
        if len(labels) > 1:
            maj = labels.most_common(1)[0][0]
            for k, v in clients.items():
                if v[0] != maj:
                    label_split[k] += 1
            divergence.append((wstart, "attack label", dict(labels)))
        views = Counter((v[3], v[4]) for v in clients.values())
        if len(views) > 1:
            maj = views.most_common(1)[0][0]
            for k, v in clients.items():
                if (v[3], v[4]) != maj:
                    stale_views[k] += 1
                    minority_view[k] = (v[3], v[4])
            divergence.append((wstart, "membership view",
                               {"majority": str(maj),
                                "minority": {k: str((v[3], v[4])) for k, v in clients.items()
                                             if (v[3], v[4]) != maj}}))

    lines = []
    def out(s=""):
        print(s); lines.append(s)

    out()
    out("=== per-instance reconciliation ===")
    out(f"{'instance':9s} {'alive':6s} {'events':7s} {'contiguous':11s} {'gap':4s} "
        f"{'mode':10s} {'n_act':5s}")
    for i in insts:
        out(f"{i.name:9s} {str(i.alive):6s} {i.events:7d} {i.contiguous:11d} "
            f"{'YES' if i.gap else '-':4s} {str(i.mode):10s} {str(i.n_active):5s}")
    out()
    out(f"gaps: {'NONE' if not any(i.gap for i in insts) else 'PRESENT'} "
        f"(each instance reconciled on its OWN contiguous seq)")
    out(f"federation mode at end: {fed_mode}")
    out(f"server instance: {len(by_name['srv'].seen)} events; "
        f"{'silence is expected in GL' if fed_mode == 'gossip' else 'active in FL'}")
    out()
    out("=== how many client instances reported the same window ===")
    for k in sorted(agree_hist):
        out(f"  {k:2d} instances: {agree_hist[k]:5d} windows")
    out(f"  windows seen: {len(window_reports)}")
    out()
    out("=== divergence (minority against the majority of reporting instances) ===")
    if not divergence:
        out("  none — every instance that reported a window agreed on the label and the "
            "membership view")
    else:
        out(f"  windows with a split: {len(divergence)} of {len(window_reports)}")
        if stale_views:
            out("  membership view — instances in the minority, and the view they hold:")
            for k in sorted(stale_views):
                out(f"      {k}: {stale_views[k]:4d} windows, view={minority_view[k]}")
            out("      a frozen view is the signature of a node ISOLATED from the "
                "federation but STILL RUNNING: isolation removes it from the pool, not "
                "from the bus")
        if label_split:
            out(f"  attack label — instances disagreeing with the majority: {dict(label_split)}")
        else:
            out("  attack label — no instance ever disagreed on WHAT the attack was")
        out("  example:")
        for w, kind, vals in divergence[:2]:
            out(f"      {w} {kind}: {vals}")
    out()
    out("=== control plane seen (by instance) ===")
    cc = Counter((nm, t) for nm, _, t, _ in control)
    for (nm, t), n in sorted(cc.items()):
        out(f"  {nm:5s} {t:22s} {n}")
    out()
    out("=== command fan-out, were the monitor to act now ===")
    for target in ("gossip", "federated"):
        p = fan_out_plan(insts, target)
        out(f"  {p['intent']:26s} reachable {p['reachable']}/{p['targets']}  "
            f"unanimity={'required' if p['unanimity_required'] else 'not required'}  "
            f"may_proceed={p['may_proceed']}")
        for k in ("abort_reason", "note"):
            if k in p:
                out(f"      {k}: {p[k]}")

    os.makedirs("results", exist_ok=True)
    path = f"results/multi_instance_monitor_{a.tag}.txt"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
