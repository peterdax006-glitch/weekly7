# B01_pit_firewall
Bible: PHASE 1 (BIBLE.md lines 432-491). Estimated code: 1,500-3,000 lines.
You own: engine/pit.py, tests/test_pit.py

Point-in-time data firewall: as-of views over price/event/insider/macro/fundamental frames with publication lags; a guard object that raises on any access after `as_of`; survivorship checks; restatement handling; audit log of every access; leak detectors (feature computed with vs without future rows must match).
