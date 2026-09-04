"""Architecture monitor — outbound observability for the adaptive federation.

An external *monitor* element observes the federation without influencing it
(strictly Level-1 / read-only, per docs/ARQUITETURA_v2.md §8). Whenever the
architecture changes (FL<->GL) or a node fails, an event is recorded with a
monotonic sequence number and UTC timestamp, appended to a persisted JSONL
audit trail, and served over a small REST endpoint the monitor *polls*.

Design choices (see the v2 design discussion):
  * Pull, not push: the monitor queries GET /events?since=<seq>; no webhook,
    no assumption that the monitor is reachable from the federation process.
  * Persisted audit trail: every event is appended to a JSONL file and the
    sequence counter continues across runs (replay-friendly).
  * Stdlib only (http.server, threading, json) — no new dependency, no dead
    weight on constrained IEDs when the monitor is disabled (NullMonitor).

Event schema (one JSON object per line in the JSONL trail, and per element of
GET /events):
    {
      "seq":          int,        # monotonic, unique, continues across runs
      "ts":           str,        # ISO-8601 UTC, microsecond precision
      "type":         str,        # "architecture_change" | "node_failure"
      "round":        int,
      "from_mode":    str | null,  # architecture_change only
      "to_mode":      str | null,  # architecture_change only
      "reason":       str | null,  # "scheduled" | "recovery" | "node_failure" | ...
      "failed_nodes": [int],       # node indices identified as failing
      "active_nodes": [int] | null,
      "detail":       str | null
    }

The architecture_change event does NOT assert causation between a failure and
the switch; it carries `recent_failed_nodes` as context, while the authoritative
per-node failure signal is the separate node_failure event.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── event store ────────────────────────────────────────────────────────────────

class EventStore:
    """Thread-safe append-only event store with a persisted JSONL trail.

    On init the store reads any existing trail to continue the sequence counter
    (monotonic across runs) and to serve historical events to the monitor.
    """

    def __init__(self, jsonl_path: str | None = None, keep_in_memory: int = 10_000) -> None:
        self._lock   = threading.Lock()
        self._events: list[dict] = []
        self._seq    = 0
        self._path   = jsonl_path
        self._keep   = keep_in_memory
        if jsonl_path:
            os.makedirs(os.path.dirname(os.path.abspath(jsonl_path)), exist_ok=True)
            self._load_existing()

    def _load_existing(self) -> None:
        if not self._path or not os.path.exists(self._path):
            return
        with open(self._path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._seq = max(self._seq, int(ev.get("seq", 0)))
                self._events.append(ev)
        # keep only the tail in memory; the file remains the full trail
        if len(self._events) > self._keep:
            self._events = self._events[-self._keep:]

    def append(self, event: dict) -> dict:
        """Assign seq + ts, persist, and store. Returns the finalized event."""
        with self._lock:
            self._seq += 1
            event = {"seq": self._seq, "ts": _utcnow(), **event}
            if self._path:
                with open(self._path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(event, ensure_ascii=False) + "\n")
                    fh.flush()
            self._events.append(event)
            if len(self._events) > self._keep:
                self._events = self._events[-self._keep:]
            return event

    def since(self, seq: int, limit: int | None = None,
              type_filter: str | None = None) -> list[dict]:
        with self._lock:
            out = [e for e in self._events
                   if e["seq"] > seq and (type_filter is None or e["type"] == type_filter)]
        return out[:limit] if limit else out

    def last_seq(self) -> int:
        with self._lock:
            return self._seq


# ── REST server (pull) ───────────────────────────────────────────────────────

class MonitorServer:
    """Minimal HTTP server the external monitor polls.

    Endpoints (all GET, JSON responses):
      /health              -> {"status": "ok"}
      /status              -> current mode, round, active/failed nodes, last_seq
      /events              -> all retained events (JSON array)
      /events?since=<seq>  -> only events with seq > since (incremental polling)
      /events?type=<t>&limit=<n>  -> optional filters
    """

    def __init__(self, store: EventStore, state: "dict",
                 host: str = "127.0.0.1", port: int = 8722) -> None:
        self._store = store
        self._state = state          # shared, updated by MonitorRecorder
        self._host, self._port = host, port
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> "MonitorServer":
        store, state = self._store, self._state

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):    # silence default stderr logging
                pass

            def _send(self, code: int, payload) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                u = urlparse(self.path)
                q = parse_qs(u.query)
                if u.path == "/health":
                    return self._send(200, {"status": "ok"})
                if u.path == "/status":
                    return self._send(200, {**state, "last_seq": store.last_seq()})
                if u.path == "/events":
                    since = int(q.get("since", ["0"])[0])
                    limit = int(q["limit"][0]) if "limit" in q else None
                    tf    = q["type"][0] if "type" in q else None
                    return self._send(200, store.since(since, limit, tf))
                return self._send(404, {"error": "not found", "path": u.path})

        self._httpd = ThreadingHTTPServer((self._host, self._port), _Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        name="monitor-server", daemon=True)
        self._thread.start()
        print(f"[Monitor] REST endpoint em http://{self._host}:{self._port} "
              f"(GET /events?since=, /status, /health)")
        return self

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()


# ── recorder (the façade HybridStrategy talks to) ──────────────────────────────

class MonitorRecorder:
    """Records architecture changes and node failures; the strategy calls this.

    Holds the current federation state (mode, round, active/failed nodes) for
    the /status endpoint, and appends every event to the audit trail.
    """

    def __init__(self, store: EventStore, server: MonitorServer | None = None,
                 state: dict | None = None) -> None:
        self._store  = store
        self._server = server
        self._state  = state if state is not None else {}
        self._state.setdefault("current_mode", None)
        self._state.setdefault("round", 0)
        self._state.setdefault("active_nodes", None)
        self._state.setdefault("failed_nodes", [])
        self._recent_failed: list[int] = []

    # -- called by HybridStrategy --------------------------------------------

    def round_start(self, round: int, mode: str, active_nodes: list[int] | None) -> None:
        # updates the live state served at /status; does NOT append an event —
        # the trail carries only architecture_change and node_failure.
        self._state.update(current_mode=mode, round=round, active_nodes=active_nodes)

    def architecture_change(self, round: int, from_mode: str, to_mode: str,
                            reason: str | None = None,
                            active_nodes: list[int] | None = None) -> dict:
        # a switch closes the current window of observed failures
        recent = sorted(set(self._recent_failed))
        reason = reason or ("node_failure" if recent else "scheduled")
        self._state.update(current_mode=to_mode, round=round, active_nodes=active_nodes)
        ev = self._store.append({
            "type": "architecture_change", "round": round,
            "from_mode": from_mode, "to_mode": to_mode, "reason": reason,
            "failed_nodes": recent, "active_nodes": active_nodes,
            "detail": f"recent_failed_nodes={recent}" if recent else None,
        })
        self._recent_failed = []
        print(f"[Monitor] architecture_change seq={ev['seq']} "
              f"{from_mode}->{to_mode} round={round} reason={reason} "
              f"failed={recent}")
        return ev

    def node_failure(self, round: int, failed_nodes: list[int],
                     detail: str | None = None) -> dict:
        failed_nodes = sorted(set(failed_nodes))
        self._recent_failed.extend(failed_nodes)
        prev = set(self._state.get("failed_nodes") or [])
        self._state["failed_nodes"] = sorted(prev | set(failed_nodes))
        ev = self._store.append({
            "type": "node_failure", "round": round, "from_mode": None,
            "to_mode": None, "reason": "node_failure",
            "failed_nodes": failed_nodes,
            "active_nodes": self._state.get("active_nodes"),
            "detail": detail,
        })
        print(f"[Monitor] node_failure seq={ev['seq']} round={round} "
              f"nodes={failed_nodes}")
        return ev

    def stop(self) -> None:
        if self._server is not None:
            self._server.stop()


# ── null object + factory ──────────────────────────────────────────────────────

class NullMonitor:
    """No-op recorder used when monitoring is disabled — zero overhead."""

    def round_start(self, *a, **k):          pass
    def architecture_change(self, *a, **k):  return None
    def node_failure(self, *a, **k):         return None
    def stop(self):                          pass


def build_monitor(conf: dict):
    """Factory — instantiates a MonitorRecorder from conf, or NullMonitor.

    conf structure (all optional; monitor is opt-in):
        monitor:
          enabled: true
          host: 127.0.0.1
          port: 8722
          trail: results/monitor_events.jsonl   # persisted audit trail
          serve: true                            # start the REST endpoint
    """
    mconf = conf.get("monitor", {}) if conf else {}
    if not mconf.get("enabled", False):
        return NullMonitor()
    store  = EventStore(jsonl_path=mconf.get("trail", "results/monitor_events.jsonl"))
    state: dict = {}
    server = None
    if mconf.get("serve", True):
        server = MonitorServer(
            store, state,
            host=mconf.get("host", "127.0.0.1"),
            port=int(mconf.get("port", 8722)),
        ).start()
    return MonitorRecorder(store, server, state)
