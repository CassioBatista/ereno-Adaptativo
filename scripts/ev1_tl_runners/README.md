# Trust-level replay runners (Paper 2, Section 5.3; protocol section 9)

These scripts ran the replay of the two Disaster-FD runs. They were kept outside the
repository while the runs were made (in `~/.audit_tmp/validate_flgl/`) and were added here
unchanged on 2026-11-25, except for the paths (working directory, `PY` override).

Assumptions: the repository at `~/ereno-Adaptativo`; the Python environment at
`~/venv-ereno314` (override with `PY=/path/to/python`); the Disaster-FD runs at
`~/datasets/disaster_fd_run1` (T1) and `~/datasets/disaster_fd_scenario2` (T2); the
evaluation-v1 data files `all_in_one_*` in the repository root (`scripts/build_ev1_datasets.py`).
One run at a time (memory); each runner refuses to start while another pipeline run is active.

| Script | What it runs | Results |
|---|---|---|
| `run_v2.py`, `v2_loader.py` | one pipeline run of a config (`ereno.py distributed`) with a lean loader of the selected columns | — |
| `run_tl_faithful.sh` + `tl_faithful_template.yaml` | faithful replay: ReSIDS, fixed FL, fixed GL, 4 domains x T1/T2 (reported) | `results/ev1_tl/faithful/` |
| `run_tl_faithful_glboth.sh` + `tl_faithful_glboth_template.yaml` | fixed GL re-run with `gl_peers: both` (same gossip nodes as ReSIDS; reported) | `results/ev1_tl/faithful/`, old runs in `faithful_gl_any/` |
| `run_tl_delta.sh`, `run_tl_delta_glboth.sh` | sensitivity to the round period, 5-600 s, ERENO and CICIDS2017 (reported) | `results/ev1_tl/delta_sens/`, old GL runs in `delta_sens_gl_any/` |
| `run_tl_faithful_seeds.sh` (uses `tl_faithful_template.yaml`) | the faithful replay with seeds 43-46, 96 runs (dispersion; exploratory), summary `scripts/ev1_tl_seeds_summary.py` | `results/ev1_tl/faithful_seeds/s<seed>/` |
| `run_tl_faithful_band15.sh` + `tl_faithful_band15_template.yaml` | ReSIDS with the gossip band at 15 (T2 reported; T1 invalid, see its `NOTE.md`) | `results/ev1_tl/faithful_band15/` |
| `run_tl_grid.sh` + `tl_grid_template.yaml` | first grid, links as seen by each monitor (record only) | `results/ev1_tl/grid_d60/` |
| `run_tl_arrival_offsets.sh` + `tl_offset_template.yaml` | arrival-offset check (record only) | `results/ev1_tl/arrival_offsets/` |

Each run is followed by `scripts/ev1_tl_node_eval.py` (per-node detection from the holdings
log); the summaries are `scripts/ev1_tl_faithful_summary.py` and
`scripts/ev1_tl_delta_summary.py`. `DRY=1` writes the configs only, into the results
directories (it overwrites the committed `cfg_*.yaml` there).

The large per-run outputs (boosters, `scores.npz`, `rounds.jsonl`, `node_eval.csv`, run logs)
are not versioned (`.gitignore`). The checks made without training, and the post hoc
diagnostics cited in the paper drafts and in the protocol change log, are in
`scripts/ev1_checks/`.
