# ReSIDS with the gossip band at 15 (t_gl: 15), 2026-11-25

- T2 runs (`*_T2_switching_gl15`): valid; no exchange across a ring link that is down.
- T1 runs (`*_T1_switching_gl15`): NOT valid, kept as record only. Between 30.5 and 44.5 min,
  GLow lets adjacent active nodes exchange across ring links that are down (GLow neighbours are
  the ring neighbours in the active set, not the neighbours over an up link).
- See docs/EVALUATION_PROTOCOL.md, change log 2026-11-25, and ~/.audit_tmp/check_down_links.py,
  ~/.audit_tmp/band15_compare.py.
