# B17_quality_pyramid
Bible: PHASES 31, 32 (lines 1631-1692). Estimated code: 1,500-3,000 lines (a requirement: land inside it with substance).
You own: scripts/quality_gate.py, scripts/test_inventory.py, tests/integration/ (new), tests/test_quality_gate.py, pyproject.toml [tool.*] sections only if absent (create setup.cfg alternative if pyproject exists)

Code-quality firewall and test pyramid: static checks (syntax, import boundaries e.g. research never imports broker order paths, no bare except in engine, no print-debug in engine, no network in tests), a test inventory mapping every Bible phase/checklist item to its tests (reports gaps), integration tests over small synthetic end-to-end runs (features -> miner -> session replay), and a single `quality_gate.py` that exits nonzero on any failure. Run it and report gaps found.
