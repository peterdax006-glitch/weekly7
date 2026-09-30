# F31_replay_determinism_fixes  [model: opus]
C75 Phase 2A/2C (deterministic replay, checkpoints, crash recovery), Firewall 5. Read CONTEXT.md (rules 1-29) and
tests/test_whole_cycle_replay.py + tests/test_whole_cycle_replay_threads.py (F30; each defect is pinned by an xfail there).
Fix at the source, then REMOVE the matching xfail / canon() exclusion so the test enforces it (a strict xfail flips to a hard failure
the moment its bug is fixed - that is the signal):
 1. REAL BUG: engine/research/loop.py:2782 _cycle_now + :2826 step - a crash after a cycle's last stage (report.cycle) but before
    state.cycle += 1 leaves all stages marked done; the next date is treated as a resumed cycle, runs no stage and is silently LOST.
    Make cycle completion atomic (e.g. the checkpoint that records the report also advances the cycle, or resume detects a completed
    cycle and moves on) - no stage may run twice and no date may be skipped. Add a hard kill test at that exact point.
 2. Thread mode is racy: loop.py:1932 st_harvest / :668 Executor.finished() harvest whatever has finished without waiting, so identical
    runs disagree on harvested counts, AUDIT counts, lineage and stage n_in/n_out. Make harvesting deterministic (harvest in submission
    order what is due by a deterministic rule, e.g. wait for the jobs submitted in the previous cycle up to a budget) without serialising
    the whole executor; the thread-mode test must pass strictly.
 3. engine/research/namespaces.py:434 ResearchStore.token hashes id(self) (a memory address) and is persisted in checkpoints - derive it
    from stable content.
 4. Provenance.created_real = datetime.now() ignoring the injected Runtime clock: frontier.py:1507, symmetry.py:1534,
    counterfactual.py:500, volatility_lab.py:1698, engine/learning/knowledge.py:675 default - route through the injected clock (keep a
    real-time default only where no clock exists, and say where).
 5. engine/learning/experiment_memory.py:969 uses current_code_hash() instead of the run's LoopConfig.code_hash - honour the configured
    hash when one is given (the registry/pre-launch path keeps engine_code).
You OWN the files named above for these changes only. Do NOT edit evidence.py, quality_gate.py, pattern_benchmark.py, calibration.py,
complexity.py, identity_firewall.py or firewall.py (F28 running), episodes.py / episode_paths.py / precursors.py (F29 running).
Never run git. <= 1 process. Run tests/test_whole_cycle_replay*.py, test_research_loop.py, test_research_dataflow.py, the tests of every
module you touch, and CI mypy. Report each fix, its test, and results.
