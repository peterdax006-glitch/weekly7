# B03_lifecycle_bank
Bible: PHASES 4 and 5 (lines 612-680). Estimated code: 1,700-3,500 lines.
You own: engine/pattern_lifecycle.py, engine/pattern_bank.py, tests/test_pattern_lifecycle.py, tests/test_pattern_bank.py

Pattern lifecycle state machine (candidate->active->watch->failed->rescoped/discarded; canon C43: never hold a failed pattern; improved form must hold long-run, discovery, confirmation and recent) with transition log; long-term bank keyed by names, versioned, earlier-windows-only reads, decay, merge of evidence across runs, file locking, integrity hash. Operates on PatternMiner.patterns frames.
