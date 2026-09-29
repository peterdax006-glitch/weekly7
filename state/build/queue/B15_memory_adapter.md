# B15_memory_adapter
Bible: PHASES 9, 14, 18 (lines 833-903, 1041-1075, 1175-1216). Estimated code: 4,200-8,500 lines (a requirement: land inside it with substance).
You own: engine/memory.py, engine/adaptive.py (you OWN these two; keep every public name/signature and keep tests/test_memory.py and tests/test_session.py passing), engine/missed_winners.py, engine/memory_diagnostics.py, tests/test_missed_winners.py, tests/test_adapter.py, tests/test_memory_ext.py

Bring the memory system, missed-winner detector and weekly adapter to the Bible's specification: every factor it lists, diagnostics (why a memory counted), memory capacity/eviction, era weighting, the missed-winner detector as its own module with evaluation (did learning from missed winners help, vs a shuffled-winners control), adapter guard rails, cooling-off, fast revert, an adaptation audit trail. Determinism is sacred: replay of the same inputs must give identical adaptations.
