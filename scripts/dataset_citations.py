#!/usr/bin/env python3
"""Citations of the requested references of the screened releases (docs/DATASET_PROTOCOL.md,
section 7), from OpenAlex: citations received in 2025 and 2026 (counts_by_year), and the
total. A reference is found by DOI when it has one, otherwise by title search, keeping the
result whose title matches; a reference not found is not indexed. The counts used by the
protocol are those recorded in DATASET_PROTOCOL.md on the counting date; re-running this
script later gives later counts.

  python scripts/dataset_citations.py
Out: results/dataset_citations.txt
"""
import datetime
import io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import redirect_stdout

YEARS = (2025, 2026)
REFS = [
    ("ERENO", "doi", "10.1109/TDSC.2023.3336857"),
    ("CICIDS2017, improved", "doi", "10.1109/CNS56114.2022.9947235"),
    ("Electra Modbus", "doi", "10.1109/ACCESS.2019.2958284"),
    ("UNSW IoT attack traces", "title", "Detecting Volumetric Attacks on IoT Devices via SDN-Based Monitoring of MUD Activity"),
    ("CICIoT2023", "doi", "10.3390/s23135941"),
    ("ToN_IoT (network)", "doi", "10.1016/j.scs.2021.102994"),
    ("IoT-23", "title", "IoT-23: A labeled dataset with malicious and benign IoT network traffic"),
    ("X-IIoTID", "doi", "10.1109/JIOT.2021.3102056"),
    ("Kitsune", "doi", "10.14722/ndss.2018.23204"),
    ("Edge-IIoTset", "doi", "10.1109/ACCESS.2022.3165809"),
    ("UAV-NIDD", "doi", "10.1109/TNSE.2025.3553442"),
    ("PDNet-IDS26", "doi", "10.5753/sbcas.2026.21576"),
    ("Hassler et al.", "doi", "10.1109/TITS.2023.3339728"),
    ("CIC-BCCC-NRC-IoMT-2024 (tabular; Sasi et al.)", "doi", "10.1016/j.jiixd.2024.09.001"),
    ("CIC IIoT 2025", "doi", "10.3390/electronics14204095"),
    ("Morris-1", "doi", "10.1109/TSG.2015.2409775"),
    ("Morris-2, Morris-3", "doi", "10.1007/978-3-662-45355-1_5"),
    ("Morris-4", "title", "Industrial Control System Simulation and Data Logging for Intrusion Detection System Research"),
    ("PowerDuck", "doi", "10.1145/3546096.3546102"),
    ("Westermo", "title", "The Westermo network traffic data set"),
    ("HAI 23.05", "title", "HAI security datasets"),
]
SEL = "display_name,cited_by_count,counts_by_year,publication_year,type"


def get(url, tries=6):
    for k in range(tries):
        try:
            return json.load(urllib.request.urlopen(url, timeout=30))
        except urllib.error.HTTPError as e:
            if e.code != 429 or k == tries - 1:
                raise
            time.sleep(5 * 2 ** k)                     # rate limit: back off and retry


def norm(s):
    return "".join(c for c in s.lower() if c.isalnum()).replace("lot", "iot")


def main():
    print(f"OpenAlex, queried {datetime.date.today().isoformat()}; window = citations received in "
          f"{YEARS[0]}-{YEARS[-1]}")
    print("release\twindow\ttotal\tyear\ttype\ttitle")
    for name, kind, key in REFS:
        if kind == "doi":
            w = get(f"https://api.openalex.org/works/https://doi.org/{key}?select={SEL}")
        else:
            r = get(f"https://api.openalex.org/works?search={urllib.parse.quote(key)}&per_page=5&select={SEL}")["results"]
            w = next((x for x in r if norm(x["display_name"] or "")[:50] == norm(key)[:50]), None)
            if w is None:
                print(f"{name}\tn.i.\tn.i.\t\t\tnot indexed (title search: "
                      + "; ".join(x["display_name"][:60] for x in r[:3]) + ")")
                time.sleep(1.5)
                continue
        win = sum(c["cited_by_count"] for c in w["counts_by_year"] if c["year"] in YEARS)
        print(f"{name}\t{win}\t{w['cited_by_count']}\t{w['publication_year']}\t{w['type']}\t{w['display_name']}")
        time.sleep(1.5)


if __name__ == "__main__":
    os.chdir(os.path.expanduser("~/ereno-Adaptativo"))
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    open("results/dataset_citations.txt", "w").write(buf.getvalue())
    print(buf.getvalue())
