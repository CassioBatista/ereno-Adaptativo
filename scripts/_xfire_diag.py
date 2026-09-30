#!/usr/bin/env python3
"""Diagnostic: per-attack cross-firing (OR of the OTHER specialists) vs Tier-2 recall.
Lets us pick a zero-day where Tier-2 genuinely fills a gap Tier-1 leaves (low cross,
high Tier-2). Not a paper figure. Out: prints a table."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import xgboost as xgb
import python.util as util
from fd.spec_detector import SpecDetector, fields_from_feature_list

FEATURES = [2,4,5,14,15,17,19,20,21,23,25,27,35,40,41,42,44,45,46,50,54,55,57,58]
SEED = 42; BENIGN_CAP = 500_000; NBR = 10
PARAMS = {"max_depth":4,"eta":0.1,"objective":"binary:logistic","seed":SEED,"nthread":4}
NAME = {1:"random_replay",2:"inverse_replay",3:"masq_fake_fault",4:"masq_fake_normal",
        5:"injection",6:"high_StNum",7:"poisoned_high_rate"}

rng = np.random.default_rng(SEED); nc = util.normal_class
Xtr_raw, ytr, _ = util.load_arff("all_in_one_ereno_train.csv")
Xtr = util.filter_features(Xtr_raw, FEATURES); del Xtr_raw
nidx = np.where(ytr==nc)[0]
if len(nidx) > BENIGN_CAP:
    drop = rng.permutation(nidx)[BENIGN_CAP:]; keep = np.ones(len(ytr),bool); keep[drop]=False
    Xtr, ytr = Xtr[keep], ytr[keep]
attacks = sorted(int(c) for c in np.unique(ytr) if c!=nc)
det = SpecDetector(fields_from_feature_list(FEATURES)).fit(Xtr[ytr==nc])
ben = np.where(ytr==nc)[0]; bslice = rng.permutation(ben)[:BENIGN_CAP//7]

spec = {}
for a in attacks:
    aidx = np.where(ytr==a)[0]
    idx = np.concatenate([aidx, bslice])
    y = np.r_[np.ones(len(aidx)), np.zeros(len(bslice))].astype(int)
    p,q = int(y.sum()), int((y==0).sum()); pr=dict(PARAMS); pr["scale_pos_weight"]=q/p
    spec[a] = xgb.train(pr, xgb.DMatrix(Xtr[idx], label=y), num_boost_round=NBR, verbose_eval=False)
del Xtr, ytr

Xte_raw, yte, _ = util.load_arff("all_in_one_ereno_test.csv"); util.normal_class=nc
Xte = util.filter_features(Xte_raw, FEATURES); del Xte_raw
pred = {a:(spec[a].predict(xgb.DMatrix(Xte))>=0.5).astype(np.int16) for a in attacks}
t2flag = det.flag(Xte)
print(f"{'attack':>20} {'own':>7} {'cross':>7} {'tier2':>7}  (cross=OR of OTHER specialists)")
for a in attacks:
    m = (yte==a); n=int(m.sum())
    own = 100.0*int((pred[a][m]==1).sum())/n
    cross_or = np.zeros(n, bool)
    for b in attacks:
        if b!=a: cross_or |= (pred[b][m]==1)
    cross = 100.0*int(cross_or.sum())/n
    t2 = 100.0*float(t2flag[m].mean())
    print(f"{NAME[a]:>20} {own:7.1f} {cross:7.1f} {t2:7.1f}")
