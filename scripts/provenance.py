#!/usr/bin/env python3
"""Provenance helpers: content hashes and the generation manifest.

A scenario file says WHAT to replay; the manifest records WITH WHAT it was replayed, so
that data + scenario + manifest reproduce a stream byte-for-byte and a third party can
verify it. Input CSVs are large (the ERENO train split is ~1.6 GB), so their digests are
cached by (path, size, mtime) in results/.input_hashes.json.
"""
import hashlib
import json
import os
import subprocess

CACHE = "results/.input_hashes.json"


def sha256(path, cache=False):
    p = os.path.realpath(path)
    st = os.stat(p)
    key = f"{p}:{st.st_size}:{int(st.st_mtime)}"
    if cache and os.path.exists(CACHE):
        try:
            c = json.load(open(CACHE, encoding="utf-8"))
            if key in c:
                return c[key]
        except (json.JSONDecodeError, OSError):
            pass
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(8 << 20), b""):
            h.update(blk)
    dig = h.hexdigest()
    if cache:
        os.makedirs("results", exist_ok=True)
        c = {}
        if os.path.exists(CACHE):
            try:
                c = json.load(open(CACHE, encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                c = {}
        c[key] = dig
        json.dump(c, open(CACHE, "w", encoding="utf-8"), indent=1)
    return dig


def file_ref(path, cache=False):
    return {"file": path, "bytes": os.stat(os.path.realpath(path)).st_size,
            "sha256": sha256(path, cache=cache)}


def git_state():
    def run(*a):
        try:
            return subprocess.run(a, capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    commit = run("git", "rev-parse", "--short", "HEAD")
    dirty = run("git", "status", "--porcelain")
    return {"commit": commit, "dirty": bool(dirty) if dirty is not None else None,
            "note": ("working tree has uncommitted changes — the commit alone does NOT "
                     "identify this generator") if dirty else None}


def write_manifest(out_path, *, scenario_path, scenario, inputs, model, stream, output,
                   generator="scripts/scenario_events.py"):
    api_version = None
    try:
        import yaml
        api_version = yaml.safe_load(open("docs/openapi.yaml", encoding="utf-8"))["info"]["version"]
    except Exception:                                   # noqa: BLE001 - provenance is best-effort
        pass
    man = {
        "generated_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(timespec="seconds"),
        "scenario": {**file_ref(scenario_path), "name": scenario["name"]},
        "generator": {**file_ref(generator), "git": git_state()},
        "schemas": {"event": file_ref("schemas/event.schema.json"),
                    "scenario": file_ref("schemas/scenario.schema.json"),
                    "api_version": api_version},
        "inputs": inputs,
        "model": model,
        "stream": stream,
        "output": output,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    json.dump(man, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    open(out_path, "a", encoding="utf-8").write("\n")
    return man
