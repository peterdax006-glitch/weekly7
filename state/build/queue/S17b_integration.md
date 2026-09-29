# S17b_integration (part B: the Test loop + API gaps S12 found)
Contract section 60 row "Integration/refactoring/type safety" (shared with S17a), sections 3, 4, 25, 64, 85, 87; canon C55-C65.
Read state/build/CONTEXT.md fully (rules 1-19), SELF_LEARNING_CONTRACT.md sections 3, 4, 25, 58-64, 83, 85, 87, canon C64
verbatim, and state/build/INTEGRATION.md (S12/S16/S19 hooks). C63: code + tests; the ONLY real-data activity allowed is the smoke in 5.
S17a is running in parallel and owns engine/lessons.py, memory.py, missed_winners.py, improve.py, registry.py, patterns.py,
ci.yml, pyproject.toml, engine/learning/curator.py (strength band) and ONE regex line in engine/learning/archive.py. Do not touch those.
You OWN: scripts/livesim_loop2.py, engine/livesim.py, engine/adaptive.py, engine/learning/memory_firewall.py,
engine/learning/knowledge.py (parent-id format only), engine/learning/context.py, engine/learning/retrieval.py,
engine/learning/similarity.py; engine/learning/archive.py put_knowledge ONLY (re-read the file right before editing - S17a edits one
regex there); new: engine/learning/test_path.py, scripts/acceptance_mini.py, tests/test_learning_test_path.py.

Do:
1. API gaps S12 reported (fix at the source, then delete S12's workarounds in learner.py only if trivially safe - else leave and
   note): memory_firewall resolves ancestry by bare id but knowledge.py writes parents as "id@vN" with a self-reference (make one
   canonical form, tested both ways); archive.put_knowledge and context.contexts_from_knowledge accept a typed ContextSet of
   Condition objects; retrieval._transfer_factor must not re-run the quadratic transfer test per row when confidence.transfer is
   None (compute once / cache); SimilarityWeights defaults must not mark every pair non-comparable at ~25% observed fields
   (coverage-aware default, tested).
2. Test path (C64 architecture) in engine/learning/test_path.py: a trusted-side runner that, per simulated day, advances the
   Curator (Curator.run_day) and feeds the trader ONLY the TraderDay plus the hardened Feed data for that day; the
   LegitimateLearner decides from that; outcomes flow back after maturity; memory is filed by real year on the trusted side.
3. Wire it into scripts/livesim_loop2.py behind a flag (--learner legit|off; default off until Stage 3 validation), keeping every
   existing gate (re-tester, future-scramble, fill audit, RevealGate, per-round checkpoint, lineage basis). Engine/livesim.py and
   adaptive.py changes only where the runner must hook in. trader_view.assert_trader_path_clean() must still pass.
4. scripts/acceptance_mini.py: reproduces S12's section-87 miniature (run_acceptance on the planted world, seeds 3 and 2 more),
   writes state/research/acceptance_mini/{report.md,summary.json} with provenance; numbers come from the run, never typed.
5. Smoke only: run livesim_loop2 with --learner legit on ONE synthetic or smallest non-THIN window for a few weeks to prove the
   wiring (not a result). RAM >= 2.5 GB rule. Do NOT stop or restart the running Test loop (PID 26844) - the main session relaunches it.
Full test suite at the end; report counts honestly.
