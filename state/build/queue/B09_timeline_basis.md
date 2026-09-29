# B09_timeline_basis
Bible: PHASES 17, 19, 20 (lines 1142-1174, 1217-1292). Estimated code: 3,200-6,500 lines.
You own: engine/timeline.py, engine/basis_search.py, engine/objective.py, tests/test_timeline.py, tests/test_basis_search.py, tests/test_objective.py

Timeline dial and yearly pacing (where are we vs the ~7%/week path; adjust aggressiveness within bounds); outer training-basis search (walk-forward over defaults+adaptation meta, screening then confirmation, anti-overfit penalty); tiered objective firewall (tier1 share of 5-10% weeks, tier2 risk, tier3 positive share; C38/C39) with property tests. Current tiered() lives in scripts/livesim_loop2.py - reimplement as a tested library.
