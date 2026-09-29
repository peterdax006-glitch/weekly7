# S16_same_year_harness
Contract sections 25 (1,300 lines), 29, 65, 66, 87; checklist E01-E07, L01-L05; canon C54, C55, C57, C58.
Read state/build/CONTEXT.md fully (rules 1-19) and SELF_LEARNING_CONTRACT.md sections 0-4, 25, 29, 58-62, 65-66, 83, 85, 87.
C63: code + unit tests now; no real-data runs.
You own: engine/learning/same_year.py, engine/learning/controls.py, tests/test_learning_same_year.py.
Build on (import, never edit): engine/learning_delta.py (make_presentation, audit_presentation, harness), engine/learners.py,
engine/antimemo.py, engine/pattern_memory.py (view(real_now,...) is the C58 date filter), engine/learning/planted_world.py
(reidentify, year_swap, reference learners), engine/learning/scorecard.py, transfer.py, learning_curve.py. B22 is still running
scripts/learning_curve*.py and owns engine/learning_delta.py - do not touch them.

Build:
1. The five frozen controls of section 25 as one interface (fit/observe/decide): A no-learning, B legitimate learner (a pluggable
   callable; default wraps pattern_memory + retrieval), C identity memoriser, D random learner (learns random updates of equal
   magnitude), E leaky learner (sees future outcomes; must be caught by the firewalls and must be distinguishable). Each control is
   frozen: code hash + config hash recorded, refusing to run if either changes (L01-L05).
2. Run 1..N of the same year, each freshly disguised (new codes, date shift, seal) with memory carried forward through the C58
   date filter; learning curve per control; the legitimate learner must beat A without resembling C (memorisation gap and
   identity gap with CIs).
3. RERUN-FINGERPRINT PROBLEM (mapping E02, contradictory): a disguised rerun shows the SAME numbers, so any numeric memory can
   recognise the year (section 1.2 forbids identifying a disguised rerun). Build and test identity-preserving perturbations that
   break exact-number recognition without destroying the patterns: random sub-universe sampling per run, per-name return scaling
   within volatility buckets, week-offset jitter, and a recognition probe (can a classifier tell run k's panel is the same year as
   run j's? report AUC; must be near 0.5 after perturbation while planted patterns stay recoverable). Quantify the information
   cost of each perturbation on planted_world.
4. Planted tests: legitimate learner improves over runs on a planted world; memoriser improves only when identities are kept and
   collapses under reidentify; random learner shows no curve; leaky learner is flagged by the future firewall; recognition probe
   AUC ~0.5 after perturbation.
