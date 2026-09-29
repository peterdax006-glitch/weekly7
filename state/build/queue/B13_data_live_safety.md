# B13_data_live_safety
Bible: PHASES 27, 28 (lines 1495-1547) + Live L2-L5. Estimated code: 1,500-3,500 lines.
You own: engine/data_sources.py, engine/isolation.py, tests/test_isolation.py, tests/test_live_safety.py, tests/test_data_sources.py

Data expansion adapters (sector ETFs, backup price source reconciliation, delisted-history registry; network code behind functions, tests use fakes); live/research isolation (research can never import broker order paths, live can never read sealed windows); paper-only and trading-hours firewall tests against engine/broker.py and engine/live.py (read them).
