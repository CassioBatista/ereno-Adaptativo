"""Loaders for audit_protocol.py: one function per audited release, returning a Release (or a
list of Releases when one release holds several tables with different schemas or labellings).

A loader only reads the release and assigns roles to its columns: label (_y), time (_t),
source address (_src), group for an independent split (_g) or the release's own test file
(_part), content features and identifiers. Identifiers are the columns that locate a row in
a capture or name an endpoint: time, addresses, ports, frame and sequence numbers,
checksums, session and patient ids. Every check runs in audit_protocol.py. Releases too
large for memory are sampled; the sample is stated in the report. Column _type, when
present, is the finest label the release provides (docs/DATASET_PROTOCOL.md, section 4);
otherwise _y is that label.
"""
import glob
import gzip
import io
import json
import os
import re
import zipfile

import numpy as np
import pandas as pd

from audit_protocol import Release

D = os.path.expanduser("~/datasets/")
LOADERS = {}


def loader(name):
    def reg(f):
        LOADERS[name] = f
        return f
    return reg


def num(df, cols):
    return df[cols].apply(pd.to_numeric, errors="coerce").astype(np.float32)


def epoch(s, **kw):
    """Seconds since 1970 from date strings; NaN where unparsable (any datetime resolution)."""
    t = pd.to_datetime(s, errors="coerce", **kw)
    if t.dt.tz is not None:
        t = t.dt.tz_convert(None)
    return (t - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)


# ------------------------------------------------------------------------- retained
@loader("ereno")
def ereno():
    import inspect_ereno as E
    tr, _, _ = E.read_arff("all_in_one_ereno_train.csv")
    te, _, _ = E.read_arff("all_in_one_ereno_test.csv")
    tr["_part"], te["_part"] = "train", "test"
    df = pd.concat([tr, te], ignore_index=True)
    del tr, te
    lab = [c for c in df.columns if c != "_part"][-1]
    ids = [E.col(i) for i in E.ABS_TIME + E.COUNTERS]
    feats = [c for c in df.columns if c not in ids + [lab, "_part"]]
    df["_y"] = df.pop(lab).astype(str)
    df["_t"] = df[E.col(1)].astype(np.float64)
    # identity fields were removed by the conversion; attribution is read from the original
    # release (train file), columns ethSrc and @class@
    p = D + "ereno_original/train.arff"
    cols, skip = [], 0
    with open(p) as f:
        for i, line in enumerate(f):
            s = line.strip()
            if s.lower().startswith("@attribute"):
                cols.append(s.split()[1])
            elif s.lower().startswith("@data"):
                skip = i + 1
                break
    att = pd.read_csv(p, header=None, names=cols, skiprows=skip, usecols=["ethSrc", "@class@"],
                      dtype=str, skip_blank_lines=True)
    att = att.rename(columns={"ethSrc": "_src", "@class@": "_y"})
    return Release("ERENO (converted release; attribution from the original)", df, feats, ids,
                   benign="normal", files=["all_in_one_ereno_train.csv", "all_in_one_ereno_test.csv"],
                   fmt="arff", attribution=att,
                   notes=["source = ethSrc of the original release (train.arff); the converted "
                          "train/test files used by the pipeline carry no identity fields"])


@loader("cicids2017c")
def cicids2017c():
    import inspect_cicids2017c as I
    z = zipfile.ZipFile(D + "cicids2017_improved/CICIDS2017_improved.zip")
    fr = []
    for d in I.DAYS:
        df = pd.read_csv(io.BytesIO(z.read(f"{d}.csv")), low_memory=False,
                         dtype={c: "str" for c in I.TEXT})
        df["_g"] = d
        fr.append(df)
    df = pd.concat(fr, ignore_index=True)
    del fr
    feats = [c for c in df.columns if c not in I.TEXT + I.IDENT + ["_g", "Attempted Category"]]
    df[feats] = num(df, feats)
    ids = I.IDENT
    for c in ("id", "Src Port", "Dst Port"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["_y"] = df["Label"].map(I.category)
    df["_type"] = df["Label"].astype(str)
    df["_t"] = epoch(df["Timestamp"], format="mixed")
    df["_src"] = df["Src IP"]
    rel = Release("CICIDS2017, improved (Liu et al., version of 27 April 2023)", df.drop(columns="_g"),
                  feats, ids, benign="Benign",
                  files=[f"zip::{z.filename}::{d}.csv" for d in I.DAYS],
                  notes=["label = category of the original label ('Attempted' flows -> Benign); "
                         "'Infiltration - Portscan' kept as its own category"])
    return rel


@loader("electra")
def electra():
    import inspect_electra as I
    p = D + "electra/electra_modbus.csv"
    df = pd.read_csv(p, dtype={c: "category" for c in I.IDENT + ["label"]}
                     | {c: "int64" for c in I.CONTENT} | {"Time": "int64"})
    df["_y"] = df.pop("label").astype(str)
    df["_t"] = df["Time"] / 1e6
    df["_src"] = df["smac"].astype(str)
    return Release("Electra Modbus", df, I.CONTENT, ["Time"] + I.IDENT, benign="NORMAL",
                   files=[p], quoted=False, notes=["all eight labels of the release"])


@loader("unsw_iot")
def unsw_iot():
    import inspect_unsw_iot as U
    X, _ = U.load()
    feats = [c for c in X.columns if not c.startswith("_")]
    X["dev"] = pd.factorize(X["_dev"])[0]
    X["ts"] = X["_ts"]
    X["_y"] = X["_cat"]
    X["_t"] = X["_ts"].astype(np.float64)
    df = X.drop(columns=["_dev", "_ts", "_cat", "_iid"] + (["_day"] if "_day" in X else []))
    return Release("UNSW IoT attack traces", df, feats, ["dev", "ts"], benign="Normal",
                   files=sorted(glob.glob(D + "unsw_iot_attack/flowdata/flowdata/*_flowstats.csv")),
                   quoted=False,
                   notes=["label = attack category of the annotated interval covering the "
                          "device-minute; counters belong to the victim device (no source)"])


# ------------------------------------------------------------------------- IoT candidates
@loader("ciciot2023")
def ciciot2023():
    import inspect_ciciot2023 as I
    parts = sorted(glob.glob(D + "ciciot2023/part-*.csv"))
    df = pd.concat([pd.read_csv(p, low_memory=False) for p in parts], ignore_index=True)
    lab = df.columns[-1]
    feats = [c for c in df.columns if c != lab]
    df[feats] = num(df, feats)
    df["_type"] = df.pop(lab).astype(str)
    df["_y"] = df["_type"].map(I.category)
    return Release("CICIoT2023", df, feats, [], benign="Benign", files=parts, quoted=False,
                   sample=f"{len(parts)} of the 169 part files")


@loader("toniot")
def toniot():
    import inspect_toniot as T
    files = sorted(glob.glob(D + "toniot/Network_dataset_*.csv"),
                   key=lambda p: int(re.findall(r"(\d+)\.csv", p)[0]))
    keep = T.IDENT + T.NUM + T.CATF + T.SESSION + ["type"]
    fr = []
    for k, p in enumerate(files):
        d = T.read(p)
        d = d[[c for c in keep if c in d.columns]].iloc[::2]       # every second flow
        d["_g"] = np.int8(k)
        fr.append(d)
    from pandas.api.types import union_categoricals
    cols = {}
    for c in fr[0].columns:
        if isinstance(fr[0][c].dtype, pd.CategoricalDtype):
            cols[c] = union_categoricals([q[c] for q in fr])
        else:
            cols[c] = np.concatenate([q[c].values for q in fr])
    del fr
    df = pd.DataFrame(cols)
    df["_y"] = df.pop("type").astype(str)
    df["_t"] = df["ts"].astype(np.float64)
    df["_src"] = df["src_ip"].astype(str)
    return Release("ToN_IoT (network part)", df.drop(columns="_g"), T.NUM + T.CATF,
                   T.IDENT + T.SESSION, benign="normal", files=files,
                   sample="every second flow of each of the 23 files")


@loader("iot23")
def iot23():
    import pickle
    import inspect_iot23 as I
    fr = []
    for p in sorted(glob.glob(D + "iot23/scan_cache/*.pkl")):
        r = pickle.load(open(p, "rb"))
        fr.append(r["sample"])
    df = pd.concat(fr, ignore_index=True)
    del fr
    for c in I.NUM + ["orig_p", "resp_p"]:
        df[c] = pd.to_numeric(df[c].replace("-", np.nan), errors="coerce").astype(np.float32)
    df["_y"] = df["dlabel"].astype(str).map(I.category)
    df["_type"] = df["dlabel"].astype(str).replace({"-": "Benign", "benign": "Benign"})
    df["_t"] = pd.to_numeric(df["ts"], errors="coerce").astype(np.float64)
    df["_src"] = df["orig_h"].astype(str)
    df["_g"] = df["_cap"]
    files = sorted(glob.glob(D + "iot23/**/conn.log.labeled", recursive=True))
    return Release("IoT-23", df, I.NUM + I.CATF, I.IDENT, benign="Benign", files=files, fmt="tsv",
                   sample=f"uniform sample of up to {I.K:,} flows per capture and detailed label "
                          "(scan cache of inspect_iot23.py)")


@loader("xiiotid")
def xiiotid():
    import inspect_xiiotid as I
    p = D + "xiiotid/xiiotid.csv"
    df = pd.read_csv(p, low_memory=False)
    flow, host, alert = I.groups(df.columns)
    feats = flow + host + alert
    for c in feats:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(str).astype("category")
    df["_y"] = df["class2"].astype(str)
    df["_type"] = np.where(df["_y"] == "Normal", "Normal", df["class1"].astype(str))
    df["_t"] = pd.to_numeric(df["Timestamp"], errors="coerce").astype(np.float64)
    df["_src"] = df["Scr_IP"].astype(str)
    for c in ("Scr_port", "Des_port"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop(columns=[c for c in I.LABELS])
    return Release("X-IIoTID", df, feats, I.IDENT, benign="Normal", files=[p],
                   notes=["label = class2 (10 classes)"])


@loader("kitsune")
def kitsune():
    import inspect_kitsune as K
    parts = K.load()
    fr = []
    for cap, Xc, yc, idx, n, _ in parts:
        d = pd.DataFrame(Xc, columns=[f"f{i}" for i in range(Xc.shape[1])])
        d["_y"] = np.where(yc == 1, cap, "normal")
        d["_t"] = idx.astype(np.float64)
        d["_cap"] = cap
        fr.append(d)
    df = pd.concat(fr, ignore_index=True)
    ids = [f"f{i}" for i in K.JIT]
    feats = [c for c in df.columns if c.startswith("f") and c not in ids]
    df["cap"] = pd.factorize(df.pop("_cap"))[0]
    files = [K.find(fd) for fd, _ in K.CAPS.values()]
    return Release("Kitsune (five captures)", df, feats, ids + ["cap"], benign="normal", files=files,
                   header=False, quoted=False, time_unit="packet",
                   sample=f"stride sample of each capture, at most {K.CAP:,} packets "
                          "(sample cache of inspect_kitsune.py)",
                   notes=["no timestamp column: _t is the packet index inside its capture; the "
                          "jitter statistics (f65-f79) carry the capture time and are treated "
                          "as identifiers; the capture id is an identifier"])


# ------------------------------------------------------------------------- first ten
@loader("edge_iiotset")
def edge_iiotset():
    import inspect_edge_iiotset as I
    p = D + "edge_iiotset/DNN-EdgeIIoT-dataset.csv"
    df = pd.read_csv(p, low_memory=False)
    ids = [c for c in I.AUTHOR_DROP + I.ARBITRARY if c in df.columns]
    feats = [c for c in df.columns if c not in ids + I.LABELS]
    for c in feats + ids:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(str).astype("category")
    df["_y"] = df["Attack_type"].astype(str)
    df["_t"] = epoch(df["frame.time"].astype(str).str.strip(), format="mixed")
    df["_src"] = df["ip.src_host"].astype(str)
    return Release("Edge-IIoTset (DNN file)", df.drop(columns=I.LABELS), feats, ids, benign="Normal",
                   files=[p], notes=["identifiers = the authors' drop list plus per-packet "
                                     "arbitrary values (sequence numbers, checksums, stream ids)"])


@loader("uav_nidd")
def uav_nidd():
    import inspect_uav_nidd as I
    out = []
    for name in ("case1_uav", "case2_ap", "case3_gcs"):
        p = os.path.join(D + "uav_nidd", I.FILES[name])
        df = pd.read_csv(p, low_memory=False, encoding="latin-1")
        lab = df.columns[-1]
        ids = [c for c in I.POSITION + I.IDENTITY if c in df.columns]
        feats = [c for c in df.columns if c not in ids + [lab]]
        for c in feats + ids:
            if not pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].astype(str).astype("category")
        df["_y"] = df.pop(lab).astype(str)
        tc = next((c for c in ("frame.time_epoch", "frame.time_relative") if c in df.columns), None)
        if tc:
            t = pd.to_numeric(df[tc].astype(str), errors="coerce").astype(np.float64)
            df["_t"] = t.where(t > 1e9)
        if "ip.src" in df.columns:
            df["_src"] = df["ip.src"].astype(str)
        ben = next((k for k in df["_y"].unique() if k.lower() in ("normal", "benign")), "Normal")
        out.append(Release(f"UAV-NIDD, {name}", df, feats, ids, benign=ben, files=[p],
                           notes=[f"label column '{lab}'"]))
    return out


@loader("pdnet_ids26")
def pdnet_ids26():
    import inspect_pdnet_ids26 as I
    p = D + "pdnet_ids26/PDNet-IDS26.csv"
    df = pd.read_csv(p, low_memory=False)
    ids = ["TCP_SrcPort", "TCP_DstPort", "TCP_Seq", "TCP_Ack"] + I.PATIENT
    feats = [c for c in df.columns if c not in ids + [I.LABEL]]
    for c in ("sex",):
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(str).astype("category")
    df["_y"] = df.pop(I.LABEL).astype(str)
    ben = next((k for k in df["_y"].unique() if k.lower() in ("normal", "benign")), "Normal")
    return Release("PDNet-IDS26", df, feats, ids, benign=ben, files=[p],
                   notes=["identifiers = TCP ports, sequence and acknowledgement numbers, and "
                          "patient identity (subject, age, sex, test time)"])


@loader("uav_hassler")
def uav_hassler():
    import inspect_uav_hassler as I
    p = D + "uav_hassler/Dataset_T-ITS.csv"
    bl = I.blocks(p)
    ident = ["timestamp_c", "timestamp_p", "frame.number", "wlan.seq", "ip.id", "tcp.seq_raw",
             "tcp.ack_raw", "time_since_last_packet"]
    schemas = {}
    for b in bl:
        schemas.setdefault(tuple(b["header"]), []).append(b["df"])
    out = []
    for k, (sig, dfs) in enumerate(schemas.items(), 1):
        df = pd.concat(dfs, ignore_index=True)
        ids = [c for c in ident if c in df.columns]
        feats = [c for c in df.columns if c not in ids + ["class"]]
        df["_y"] = df.pop("class").astype(str)
        tc = next((c for c in ("timestamp_c", "timestamp_p") if c in df.columns), None)
        if tc:
            df["_t"] = df[tc].astype(np.float64)
        ben = next((v for v in df["_y"].unique() if re.search("normal|benign", v, re.I)), "(none)")
        kind = "cyber" if "frame.len" in sig else "physical"
        out.append(Release(f"Hassler et al., schema S{k} ({kind})", df, feats, ids, benign=ben,
                           files=[p] if k == 1 else [], fmt="csv",
                           notes=[f"{len(dfs)} block(s) of the stacked CSV share this schema"]))
    return out


@loader("iomt2024")
def iomt2024():
    import inspect_iomt2024 as I
    files = sorted(glob.glob(D + "cic_bccc_nrc_iomt2024/*.csv"))
    fr = []
    for p in files:
        d = pd.read_csv(p, low_memory=False, encoding="latin-1",
                        dtype={c: "str" for c in I.TEXT})
        d["_file"] = os.path.basename(p)[:-4]
        fr.append(d)
    df = pd.concat(fr, ignore_index=True)
    del fr
    feats = [c for c in df.columns if c not in I.TEXT + I.IDENT + ["_file"]]
    df[feats] = num(df, feats)
    for c in ("Src Port", "Dst Port"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["_type"] = df["_file"].where(df["_file"] != "Benign Traffic", "Benign")
    df["_y"] = df.pop("_file").map(I.category)
    df["_t"] = epoch(df["Timestamp"], format="%d/%m/%Y %I:%M:%S %p")
    df["_src"] = df["Src IP"]
    return Release("CIC-BCCC-NRC-IoMT-2024", df.drop(columns=["Attack Name", "Label"]), feats, I.IDENT,
                   benign="Benign", files=files,
                   labelled_files=[(p, I.category(os.path.basename(p)[:-4])) for p in files],
                   notes=["label = category of the file (one CSV per class)"])


@loader("cic_iiot2025")
def cic_iiot2025():
    import inspect_cic_iiot2025 as I
    out = []
    for w in ("1sec", "10sec"):
        ps = [os.path.join(D + "cic_iiot2025", f"{k}_samples_{w}.csv") for k in ("benign", "attack")]
        df = pd.concat([pd.read_csv(p, low_memory=False) for p in ps], ignore_index=True)
        lists, feats = I.split_cols(df)
        df[feats] = num(df, feats)
        df["_y"] = df["label2"].astype(str)
        # label4 = category and attack type (83); label_full adds the target of the window
        df["_type"] = np.where(df["_y"] == "benign", "benign", df["label4"].astype(str))
        df["_t"] = epoch(df["timestamp_start"], utc=True)
        df["dev"] = pd.factorize(df["device_name"])[0]
        df["tstart"] = df["_t"]
        out.append(Release(f"CIC IIoT 2025 (DataSense), {w} windows",
                           df.drop(columns=[c for c in I.IDCOLS + lists if c in df.columns]),
                           feats, ["dev", "tstart"], benign="benign", files=ps,
                           notes=["label = label2 (category); list-valued columns left out; "
                                  "a window describes one device: no source address"]))
    return out


@loader("morris1")
def morris1():
    files = sorted(glob.glob(D + "morris_power/data*.csv"), key=lambda p: int(os.path.basename(p)[4:-4]))
    fr = []
    for p in files:
        d = pd.read_csv(p, low_memory=False)
        d["_g"] = int(os.path.basename(p)[4:-4])
        fr.append(d)
    df = pd.concat(fr, ignore_index=True)
    feats = [c for c in df.columns if c not in ("marker", "_g")]
    df[feats] = num(df, feats).replace([np.inf, -np.inf], np.nan)
    df["_y"] = df.pop("marker").astype(str)
    return Release("Morris-1 (power system, binary set)", df, feats, [], benign="Natural",
                   files=files, quoted=False,
                   notes=["15 sets, each a 1% random sample of the scenario recordings; "
                          "independent split = held-out sets"])


def _morris(title, f, lab, clock):
    import inspect_morris_ics as I
    df = I.load_arff(os.path.join(D + "morris_ics", f))
    y = df[lab].astype(int).map(I.NAMES)
    typ = (np.where(y == "normal", "normal", "specific " + df["specific result"].astype(str))
           if "specific result" in df else y.values)
    df = df.drop(columns=[c for c in ("binary result", "categorized result", "specific result", "result")
                          if c in df])
    ids = ["time"] + [c for c in df.columns if "crc" in c]
    feats = [c for c in df.columns if c not in ids]
    df["_y"] = y.values
    df["_type"] = typ
    if clock:
        df["_t"] = df["time"].astype(np.float64)
    return Release(title, df, feats, ids, benign="normal", files=[os.path.join(D + "morris_ics", f)],
                   fmt="arff", notes=["the Modbus address field names the slave, the same for "
                                      "normal and attack traffic: no emitter address"]
                   + ([] if clock else ["the 'time' column is not a clock (values 1.0-1.3): no "
                                        "time order"]))


@loader("morris2")
def morris2():
    return _morris("Morris-2 (gas pipeline, 2014)", "gas_final.arff", "result", False)


@loader("morris3")
def morris3():
    return _morris("Morris-3 (water tank, 2014)", "water_final.arff", "result", False)


@loader("morris4")
def morris4():
    return _morris("Morris-4 (gas pipeline, 2015)", "IanArffDataset.arff", "categorized result", True)


@loader("powerduck")
def powerduck():
    base = D + "powerduck/dataset/"
    rows = []
    files = sorted(glob.glob(base + "ipal/*/*.ipal.gz"))
    for p in files:
        kind = p.split("/")[-2]
        scen = os.path.basename(p).replace(".ipal.gz", "")
        fam = "normal" if kind == "Normal" else re.sub(r"^\d+-", "", scen).split("-")[0]
        with gzip.open(p, "rt") as f:
            for line in f:
                try:
                    j = json.loads(line)
                except ValueError:
                    continue
                dat = j.get("data") or {}
                r = {"_scen": scen, "_fam": fam, "id": j["id"], "t": j["timestamp"],
                     "mal": bool(j["malicious"]), "src": j["src"], "dst": j["dest"],
                     "length": j["length"], "n_values": len(dat)}
                for k, v in dat.items():
                    if isinstance(v, (int, float, bool)):
                        r[f"d_{k}"] = float(v)
                rows.append(r)
    df = pd.DataFrame(rows)
    del rows
    dcols = [c for c in df.columns if c.startswith("d_")]
    ids = ["id", "t", "src", "dst"] + [c for c in ("d_stNum", "d_sqNum") if c in df.columns]
    feats = ["length", "n_values"] + [c for c in dcols if c not in ids]
    for c in ("src", "dst"):
        df[c] = df[c].astype(str).astype("category")
    df["_y"] = np.where(df["mal"], df["_fam"], "benign")
    df["_type"] = np.where(df["mal"], df["_scen"], "benign")
    df["_t"] = df["t"].astype(np.float64)
    df["_src"] = df["src"].astype(str)
    df["_g"] = df["_scen"]
    return Release("PowerDuck", df.drop(columns=["mal", "_scen", "_fam"]), feats, ids, benign="benign",
                   files=files, fmt="jsonl-gz",
                   notes=["label = attack family of the recording for malicious packets, benign "
                          "otherwise; content = packet length, number of data fields and the "
                          "numeric data values; independent split = held-out recordings"])


@loader("westermo")
def westermo():
    labs = ["IT_B_Label", "IT_M_Label", "NST_B_Label", "NST_M_Label"]
    ident = ["sAddress", "rAddress", "sMACs", "rMACs", "sIPs", "rIPs", "startDate", "endDate",
             "start", "end", "startOffset", "endOffset"]
    files = [D + f"westermo/extended/flows/output_{s}.csv" for s in ("bottom", "left", "right")]
    base = pd.concat([pd.read_csv(p, low_memory=False) for p in files], ignore_index=True)
    feats = [c for c in base.columns if c not in labs + ident + ["protocol"]]
    base[feats] = num(base, feats)
    ids = [c for c in ident if c in base.columns]
    for c in ids:
        if not pd.api.types.is_numeric_dtype(base[c]):
            base[c] = base[c].astype(str).astype("category")
    out = []
    for lab, title in (("IT_M_Label", "timing labelling"), ("NST_M_Label", "attacker-involvement labelling")):
        df = base.drop(columns=labs).copy()
        df["_y"] = base[lab].astype(str)
        df["_t"] = pd.to_numeric(base["start"], errors="coerce").astype(np.float64)
        df["_src"] = base["sIPs"].astype(str)
        ben = df["_y"].value_counts().index[0]
        out.append(Release(f"Westermo, {title} ({lab})", df, feats, ids, benign=ben,
                           files=files if lab == "IT_M_Label" else []))
    return out


@loader("hai2305")
def hai2305():
    import inspect_hai2305 as I
    d = D + "hai_2305"
    names = ["hai-train1", "hai-train2", "hai-train3", "hai-train4", "hai-test1", "hai-test2"]
    fr = {f: pd.read_csv(os.path.join(d, f + ".csv")) for f in names}
    man = I.manual_table(d)
    for k, prefix in (("1", "A1"), ("2", "A2")):
        df = fr[f"hai-test{k}"]
        df["label"] = pd.read_csv(os.path.join(d, f"label-test{k}.csv"))["label"].values
        m = man[man["id"].str.startswith(prefix)].reset_index(drop=True)
        df["loop"] = "normal"
        for i, (s, e) in enumerate(I.segments(df["label"])):
            if i < len(m):
                df.loc[s:e, "loop"] = m.iloc[i]["loop"]
        df.drop(columns="label", inplace=True)
    for f in names[:4]:
        fr[f]["loop"] = "normal"
    for f, df in fr.items():
        df["_part"] = "test" if f == "hai-test2" else "train"
    df = pd.concat(fr.values(), ignore_index=True)
    pv = [c for c in df.columns if c not in ("timestamp", "loop", "_part")]
    df["_y"] = df.pop("loop")
    df["_t"] = epoch(df["timestamp"])
    return Release("HAI 23.05", df.drop(columns="timestamp"), pv, [], benign="normal",
                   files=[os.path.join(d, f + ".csv") for f in names],
                   notes=["label = target control loop of the attack (joined from the technical "
                          "manual), normal otherwise; no network traffic: no source address; "
                          "independent split = hai-test2 held out (trained on the train files "
                          "and hai-test1)"])
