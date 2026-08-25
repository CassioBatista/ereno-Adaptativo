#!/usr/bin/env python3
"""Latencia de feature-extraction e inferencia (seed 42, specialist, N=10).
Mede: (a) extracao de features (filter) us/amostra; (b) inferencia batch (10 boosters +
k-de-n) us/amostra amortizado (throughput); (c) inferencia single-sample us/amostra
(streaming realista). Threshold tau=0.5.
Saida: results/inference_latency.csv
Rodar: ~/venv-ereno314/bin/python scripts/inference_latency.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import python.util as util
from fd.dataset import load_and_partition

DATASET, TESTFILE = "all_in_one_ereno_train", "all_in_one_ereno_test"
FEATURES = [2, 4, 5, 14, 15, 17, 19, 20, 21, 23, 25, 27, 35, 40, 41, 42, 44, 45, 46, 50, 54, 55, 57, 58]
N, SEED = 10, 42
BATCH = 100_000     # amostras p/ throughput
SINGLE = 2_000      # amostras p/ latencia single-sample


def main():
    print("[lat] treinando 10 especialistas (seed 42)...")
    parts, _, _ = load_and_partition(f"{DATASET}.csv", FEATURES, N, seed=SEED,
                                     partitioner="attack", benign_cap=500000)
    nc = util.normal_class
    fl = []
    for X_c, y_c in parts:
        Xct, _, yct, _ = train_test_split(X_c, y_c, test_size=0.2, random_state=SEED)
        yy = (yct != nc).astype(int)
        p, q = int(yy.sum()), int((yy == 0).sum())
        pr = {"max_depth": 4, "eta": 0.1, "objective": "binary:logistic", "seed": SEED, "nthread": 1}
        if p and q: pr["scale_pos_weight"] = q / p
        fl.append(xgb.train(pr, xgb.DMatrix(Xct, label=yy), num_boost_round=10, verbose_eval=False))

    print("[lat] carregando teste (raw, sem filtrar)...")
    Xte_raw, yte_m, _ = util.load_arff(f"{TESTFILE}.csv")
    util.normal_class = nc

    # (a) feature extraction: filter_features sobre BATCH amostras raw
    idx = np.random.default_rng(SEED).permutation(len(yte_m))[:BATCH]
    raw = Xte_raw[idx]
    t0 = time.perf_counter()
    Xf = util.filter_features(raw, FEATURES)
    t_feat = (time.perf_counter() - t0) / BATCH * 1e6      # us/amostra

    # (b) inferencia batch (throughput amortizado): DMatrix + predict 10 + k-de-n
    dte = xgb.DMatrix(Xf)
    t0 = time.perf_counter()
    P = np.column_stack([b.predict(dte) for b in fl])
    votes = (P >= 0.5).sum(1)
    _ = (votes >= 1); _ = (votes >= 2)
    t_batch = (time.perf_counter() - t0) / BATCH * 1e6     # us/amostra

    # (c) inferencia single-sample (streaming): 1 amostra por vez
    Xs = Xf[:SINGLE]
    lat = []
    for i in range(SINGLE):
        d1 = xgb.DMatrix(Xs[i:i + 1])
        t0 = time.perf_counter()
        p = np.array([b.predict(d1)[0] for b in fl])
        v = int((p >= 0.5).sum()); _ = v >= 2
        lat.append((time.perf_counter() - t0) * 1e6)       # us
    lat = np.array(lat)

    print("\n===== LATENCY (seed 42, N=10 specialists, k-de-n fusion) =====")
    print(f"(a) feature extraction:        {t_feat:8.2f} us/sample")
    print(f"(b) inference, batched:        {t_batch:8.2f} us/sample  (throughput "
          f"{1e6/t_batch:,.0f} samples/s)")
    print(f"(c) inference, single-sample:  {lat.mean():8.2f} us/sample "
          f"(median {np.median(lat):.2f}, p95 {np.percentile(lat,95):.2f})")
    print(f"    end-to-end single (feat+infer) approx: {t_feat + lat.mean():.2f} us/sample")

    os.makedirs("results", exist_ok=True)
    with open("results/inference_latency.csv", "w") as fh:
        fh.write("metric,us_per_sample\n")
        fh.write(f"feature_extraction,{t_feat:.4f}\n")
        fh.write(f"inference_batched,{t_batch:.4f}\n")
        fh.write(f"inference_single_mean,{lat.mean():.4f}\n")
        fh.write(f"inference_single_median,{np.median(lat):.4f}\n")
        fh.write(f"inference_single_p95,{np.percentile(lat,95):.4f}\n")
    print("\n[lat] CSV: results/inference_latency.csv")


if __name__ == "__main__":
    main()
