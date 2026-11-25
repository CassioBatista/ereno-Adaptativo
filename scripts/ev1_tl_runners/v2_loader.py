"""Lean drop-in for python.util.load_arff for this 9.9 GB VM, any evaluation-v1 domain:
reads only the given 1-based columns (and the class) as float32, NaN kept (Morris-4 fields
absent from a message type are missing values, as in scripts/ev1_step1.py), class values
unquoted; the other columns are left at 0 (util.filter_features never reads them).
Returns (X, y, classes) and sets util.normal_class = y[0], like util.load_arff."""
import numpy as np


def make_loader(keep):
    keep = sorted(int(f) for f in keep)

    def load(path):
        import pandas as pd
        import python.util as util
        classes, skip, nattr = None, 0, 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                skip += 1
                s = line.strip()
                if s.lower().startswith("@attribute"):
                    nattr += 1
                    if "{" in s:
                        classes = [c.strip().strip("'\"") for c in s[s.index("{") + 1:s.rindex("}")].split(",")]
                elif s.lower().startswith("@data"):
                    break
        cols = [f - 1 for f in keep]
        cmap = {c: i for i, c in enumerate(classes)}
        Xs, ys = [], []
        for ch in pd.read_csv(path, skiprows=skip, header=None, usecols=cols + [nattr - 1],
                              chunksize=500_000, skip_blank_lines=True):
            Xs.append(ch[cols].to_numpy(np.float32))
            yy = ch[nattr - 1].astype(str).str.strip().str.strip("'\"").map(cmap)
            assert not yy.isna().any(), f"undeclared class in {path}"
            ys.append(yy.to_numpy(np.int64))
        Xsel, y = np.concatenate(Xs), np.concatenate(ys)
        del Xs
        X = np.zeros((len(y), nattr - 1), np.float32)
        X[:, cols] = Xsel
        util.normal_class = int(y[0])
        print(f"[v2_loader] {path}: {len(y):,} rows, {len(cols)} of {nattr - 1} columns")
        return X, y, classes

    return load
