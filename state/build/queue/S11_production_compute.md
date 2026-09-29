# S11_production_compute
Contract sections 44, 45, 57, 58. Checklist J01-J07, J09, J11 support. Minimum lines: 700+700+900+700 = 3,000.
Read state/build/CONTEXT.md fully (rules 1-18) and SELF_LEARNING_CONTRACT.md sections 0-4, 58-62, 83, 85 plus the sections below, in full. C63: foundation first - all code + unit tests now, no real-data runs or tuning.
You own: engine/learning/promotion.py, champion.py, compute.py, checkpoints.py; tests/test_learning_promotion.py, test_learning_compute.py.

Build: Champion/challenger/shadow/retired knowledge with replacement only after OOS, transfer, risk, anti-memorisation, stability and reproducibility. Promotion gate over all ten section-45 gates where one critical failure blocks, with a written rejection report. Compute manager: deterministic isolated workers, memory snapshots, checkpointing, crash and OOM recovery, stale-code detection, result reconciliation, duplicate prevention. Continuous-execution checkpoints (save state, next action, failures, current experiment, code hash; resume). Build on engine/champion.py, engine/checkpoint.py, engine/resources.py, engine/registry.py.
