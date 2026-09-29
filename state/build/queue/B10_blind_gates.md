# B10_blind_gates
Bible: PHASES 21, 22, 23, 24 (lines 1293-1412). Estimated code: 3,700-7,500 lines.
You own: engine/blind_gates.py, engine/retester.py, engine/health.py, tests/test_blind_gates.py, tests/test_retester.py, tests/test_health.py

Blind-simulator hardening gates (disguise checks, sealed-window integrity, clock never ahead), re-tester (rerun vs live within 0.5%, FIRST compare provenance code_hash: a mismatch is STALE CODE, not a parity failure - see engine/provenance.py), worker health (timeouts, memory ceilings, heartbeat, crash classification, exclusion report).
