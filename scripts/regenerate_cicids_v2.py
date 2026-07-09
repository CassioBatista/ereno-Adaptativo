"""Regenera o all_in_one_cicids_v2.csv a partir dos CSVs originais do CICIDS2017.

O arquivo v2 (392 MB) NAO e versionado no git (limite de 100 MB do GitHub);
este script e a receita reprodutivel. Ver docs/DATASETS.md para a proveniencia.

Fonte dos originais (formato MachineLearningCVE, 78 features + Label):
  oficial : https://www.unb.ca/cic/datasets/ids-2017.html  (via formulario)
  espelho : https://huggingface.co/datasets/c01dsnap/CIC-IDS2017

Uso:
    python scripts/regenerate_cicids_v2.py <dir_com_os_3_csvs> [saida.csv]

Critério (documentado na tese):
  - 3 dias do CICIDS2017: Tuesday (FTP/SSH-Patator), Wednesday (4x DoS +
    Heartbleed) e Thursday-Morning (ataques web) — os dias que contêm as
    11 classes do all_in_one original;
  - SEM subamostragem por classe (o all_in_one v1 aplicava teto de ~10k
    em BENIGN e hulk, invertendo a prevalência para 83% de ataques);
  - higiene: remove linhas com NaN/Infinity (~0,13%);
  - mesmas 78 features, mesma ordem (F1=Destination Port ... F78=Idle Min),
    mesmo formato ARFF e mesma ordem de classes do v1 — drop-in replacement.
"""

import os
import sys

import numpy as np
import pandas as pd

FILES = [
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
]

CLASS_ORDER = ["BENIGN", "Heartbleed", "GoldenEye", "Slowhttptest", "hulk",
               "slowloris", "xss", "ssh", "brute", "ftp", "sql"]

LABEL_MAP = {
    "BENIGN": "BENIGN", "Heartbleed": "Heartbleed",
    "DoS GoldenEye": "GoldenEye", "DoS Slowhttptest": "Slowhttptest",
    "DoS Hulk": "hulk", "DoS slowloris": "slowloris",
    "SSH-Patator": "ssh", "FTP-Patator": "ftp",
}


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "all_in_one_cicids_v2.csv"

    frames = []
    for f in FILES:
        path = os.path.join(src, f)
        if not os.path.exists(path):
            sys.exit(f"Arquivo original ausente: {path}")
        df = pd.read_csv(path, encoding="latin1", low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        frames.append(df)
        print(f"lido: {f} ({len(df):,} linhas)")
    orig = pd.concat(frames, ignore_index=True)
    orig["Label"] = orig["Label"].str.strip()

    label_map = dict(LABEL_MAP)
    for lb in orig["Label"].unique():        # rotulos web tem encoding variavel
        if "XSS" in lb:
            label_map[lb] = "xss"
        elif "Brute" in lb:
            label_map[lb] = "brute"
        elif "Sql" in lb or "SQL" in lb:
            label_map[lb] = "sql"
    orig["classe"] = orig["Label"].map(label_map)
    unknown = orig.loc[orig["classe"].isna(), "Label"].unique()
    if len(unknown):
        sys.exit(f"Rotulos sem mapeamento: {unknown}")

    feat_cols = [c for c in orig.columns if c not in ("Label", "classe")]
    assert len(feat_cols) == 78, f"esperava 78 features, achei {len(feat_cols)}"

    num = orig[feat_cols].apply(pd.to_numeric, errors="coerce")
    bad = num.isna().any(axis=1) | np.isinf(num.to_numpy()).any(axis=1)
    clean = orig.loc[~bad].reset_index(drop=True)
    print(f"higiene: {bad.sum():,} linhas com NaN/Inf removidas "
          f"({bad.sum() / len(orig):.2%}) -> {len(clean):,} restantes")

    X = clean[feat_cols].to_numpy(dtype=np.float64)
    y = clean["classe"].to_numpy()

    with open(out, "w", newline="\n") as fh:
        fh.write("@relation testTraffic\n\n")
        for i in range(1, 79):
            fh.write(f"@attribute F{i} numeric\n")
        fh.write("@attribute @class@ {" + ", ".join(CLASS_ORDER) + "}\n\n@data\n")
        for row, lab in zip(X, y):
            fh.write(",".join(np.format_float_positional(v, trim="-") for v in row))
            fh.write(f",{lab}\n")

    print(f"\ngerado: {out} ({os.path.getsize(out) / 1e6:.0f} MB)")
    vc = pd.Series(y).value_counts()
    for c in CLASS_ORDER:
        n = int(vc.get(c, 0))
        print(f"  {c:<14} {n:>10,} ({n / len(y):.2%})")


if __name__ == "__main__":
    main()
