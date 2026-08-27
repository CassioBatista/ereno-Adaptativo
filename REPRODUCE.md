# Reproducing ReSIDS

End-to-end reproduction of the ReSIDS results from a **clean environment**, with a
**single command**, a signed **manifest** (`MANIFEST.json`), input **SHA-256 hashes**,
and a **TP/FP/TN/FN audit**. Archived at Zenodo `10.5281/zenodo.22084923`.

## 1. Clean environment

```bash
python -m venv venv                 # Python 3.14.x
venv/bin/pip install -r requirements.lock.txt
```

`requirements.lock.txt` pins exact versions (xgboost 3.3.0, flwr 1.32.0, ray 2.55.1,
scikit-learn 1.9.0, numpy 2.5.0, scipy 1.18.0, matplotlib 3.11.0).

## 2. Minimal run (single command)

```bash
venv/bin/python reproduce.py            # min-run (minutes; hashes small inputs)
venv/bin/python reproduce.py --strict   # also hash the two 1.7 GB CSVs
```

The min-run:

1. verifies input hashes against `MANIFEST.json`;
2. loads the persisted GRASP-24 feature manifest (**no GRASP re-run, no distributed
   simulation**);
3. trains the ten specialists under the fixed seeds (data 42, GRASP 5, evaluation 7);
4. evaluates the **full held-out ERENO test set**;
5. writes to `results/`:
   - `reproduce_table.csv` — headline metrics (OR `k>=1` and corroboration `k>=2`),
   - `reproduce_confusion.txt` — **TP/FP/TN/FN audit** + per-attack recall,
   - `reproduce_figure.png` — per-attack recall (OR vs `k>=2`);
6. checks the headline numbers against `MANIFEST.expected_min_run` and prints PASS/FAIL.

**Acceptance target (reference seed 42):** F1 = 95.72 at 17,881 false positives;
confusion TP 200,444 · FP 17,881 · FN 65 · TN 2,737,258.

## 3. Heavy stages (NOT part of the min-run)

| Stage | Command | Cost |
|---|---|---|
| Dataset assembly (~1.7 GB) | `scripts/build_ereno_dataset.py` | one-off |
| GRASP feature selection | `python ereno.py grasp GR-G-VND 6 all_in_one_ereno_train --sample 150000` | ~3.5 h/run |

These reproduce the persisted `features/*.json`; the min-run consumes their output.

## 4. Provenance

Every table/figure in the paper maps, in `MANIFEST.json` (`provenance`), to a result
file with its SHA-256 and producing script. All such files post-date the corrected
pipeline freeze (2026-07-16: content-dedup union, held-out test, equal data budgets);
no number is carried over from any earlier draft.
