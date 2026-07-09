"""Converte o ERENO IEC-61850 (Kaggle sequincozes/ereno-iec61850-ids) para o
formato do pipeline: features numericas F1..FN + classe, em ARFF.

Decisao de projeto (ver docs/DATASETS.md): os ~11 atributos NOMINAIS do
original (ethDst/ethSrc/ethType, gooseAppid, TPID, gocbRef, datSet, goID,
test, ndsCom, protocol) sao DESCARTADOS:
  - identidades (MACs, goID) sao vazamento em testbed sintetico — o modelo
    memorizaria "quem" ataca, nao "como";
  - o sinal comportamental ja esta nas features numericas derivadas
    (stDiff, sqDiff, timestampDiff, delay, ...);
  - um IDS independente de identidades generaliza melhor (MAC e forjavel).

Uso:
    python scripts/build_ereno_dataset.py <train.arff> <saida_train.csv>
    python scripts/build_ereno_dataset.py <test.arff>  <saida_test.csv>
"""

import sys


def parse_header(fh):
    """Le o cabecalho ARFF; retorna (indices_numericos, classes)."""
    numeric_idx, classes, i = [], None, 0
    for line in fh:
        s = line.strip()
        low = s.lower()
        if low.startswith("@attribute"):
            if "@class@" in s:
                classes = s[s.index("{") + 1:s.index("}")].replace(" ", "").split(",")
            else:
                if low.split()[-1] == "numeric":
                    numeric_idx.append(i)
                i += 1
        elif low.startswith("@data"):
            return numeric_idx, classes
    raise ValueError("cabecalho ARFF sem @data")


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]

    with open(src, encoding="latin1") as fin:
        numeric_idx, classes = parse_header(fin)
        n_feat = len(numeric_idx)
        print(f"{src}: {n_feat} features numericas mantidas, "
              f"classes: {classes}")

        with open(dst, "w", newline="\n") as fout:
            fout.write("@relation erenoIEC61850\n\n")
            for i in range(1, n_feat + 1):
                fout.write(f"@attribute F{i} numeric\n")
            fout.write("@attribute @class@ {" + ", ".join(classes) + "}\n\n@data\n")

            kept, skipped = 0, 0
            for line in fin:
                if not line.strip():
                    continue
                parts = line.rstrip("\n").split(",")
                try:
                    vals = [parts[j].strip() for j in numeric_idx]
                    label = parts[-1].strip()
                    # valida numericos (descarta linhas corrompidas/NaN/Inf)
                    ok = all(v and v.lower() not in ("nan", "inf", "-inf", "infinity")
                             for v in vals)
                    float_probe = [float(v) for v in vals]  # noqa: F841
                    if not ok or label not in classes:
                        skipped += 1
                        continue
                except (ValueError, IndexError):
                    skipped += 1
                    continue
                fout.write(",".join(vals) + "," + label + "\n")
                kept += 1

    print(f"{dst}: {kept:,} linhas gravadas, {skipped:,} descartadas")


if __name__ == "__main__":
    main()
