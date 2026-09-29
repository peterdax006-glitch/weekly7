# S18_leak_audit_recompute
Contract section 30, 55, 85 PRIORITY 1; checklist H02, H03, H12, L13 (future-information audit passed), K14. Canon C56.
Read state/build/CONTEXT.md fully (rules 1-19) and SELF_LEARNING_CONTRACT.md sections 28-30, 55, 59, 82, 85.
You own: engine/leak_audit.py and scripts/leak_audit.py (EXCEPTION to rule 1: you may edit these two existing files; nobody else
touches them), tests/test_leak_audit_verdicts.py, state/research/leak_audit/.

Problem (found 29 Sep): state/research/leak_audit/report.md (written 00:39) says 3 LEAK channels (2 back-adjusted prices, 4 learned
state from the future, 8c feed shape), but the default blind path changed at 00:55: engine/livesim.py Feed defaults to
tradable_rule="split_invariant" and blind_feed_class(hardened=True); scripts/livesim_loop2.py plays each window with a basis version
chosen from its BasisLineage (plan[r]["version"] / "untrained"). Worse, scripts/leak_audit.py assemble() TYPES some verdicts as
literals (channel 2 is always L.LEAK). A verdict must be COMPUTED from the code and data it describes, never typed.

Do:
1. Make every channel verdict computed: inspect the actual default path (AST / call-site checks on livesim_loop2.py and
   engine/livesim.py defaults, plus the existing measurements) and derive LEAK / FIXED / QUARANTINED / CLEAN. A channel is FIXED
   only if the default path provably uses the fix AND a planted-leak test proves the fixed path would catch/neutralise it.
2. Channel 4 specifically: verify that EVERY basis a played window uses was trained only on windows that ended before that window
   started (lineage check over the real loop state file, trusted side only, never reading sealed contents), and that the basis
   TRAINING call (train_basis at livesim_loop2.py ~419, no as_of) cannot leak into a play. If training on all archived windows
   still reaches any play, it is LEAK - say exactly where.
3. Tests: for each channel, flipping the default back to the unsafe setting must flip the computed verdict to LEAK (a check that
   cannot fail is worthless). Test the empty case (missing part files -> UNMEASURED, never CLEAN).
4. Re-run the audit detached on the real caches (RAM >= 2.5 GB rule, checkpointed parts), regenerate report.md/summary.json, and
   report the new verdict table. Honest: if something is still LEAK, it stays LEAK.
This is code-plus-audit (C63 allows it: an audit of existing code is not tuning).
