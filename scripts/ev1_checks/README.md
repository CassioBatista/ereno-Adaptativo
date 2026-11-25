# Checks and post hoc diagnostics (Paper 2)

Read-only scripts cited in the paper drafts and in the protocol change log. They were kept in
`~/.audit_tmp/` while the work was done and were added here on 2026-11-25 unchanged. They
assume the repository at `~/ereno-Adaptativo`, the Disaster-FD runs at `~/datasets/`, and,
for the replay checks, the per-run outputs of `results/ev1_tl/` (not versioned).

Replay, without training (the trace and the policy only):
- `check_gl_peers.py`: gossip nodes of ReSIDS vs fixed GL with `gl_peers` any and both.
- `check_fl_nodes.py`: federated rounds of ReSIDS vs the nodes of fixed FL.
- `check_band15.py`: gossip nodes of ReSIDS with the band at 15 vs fixed GL any.
- `check_down_links.py`: exchanges across ring links that are down, per policy.

Replay, on the per-run outputs:
- `ereno_t1_block30.py`, `ereno_t2_protect.py`, `faithful_endnodes.py`: protection per node.
- `band15_compare.py`, `band15_t1_check.py`: gossip band at 15 against 30 and fixed modes.

Windows (Section 5.4):
- `win_by_volume.py`: alarms by number of attack samples in the window.
- `win_attack_removed.py`: windows that held attacks, with the attack samples removed.
- `cadence_fa_same.py`: the false alarm of the fixed-threshold cadence at 1-10 s.

Detection model (Sections 5.1 and 5.2):
- `crossdetect_counts.py`, `pertype_table_one.py`, `seed_dispersion.py`,
  `morris4_test_identical.py`, `morris4_fp_cause.py`.
