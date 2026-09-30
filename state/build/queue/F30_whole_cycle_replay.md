# F30_whole_cycle_replay  [model: sonnet]
C75 Phase 2A/2C ("deterministic replay", "checkpoints", "crash recovery") and gap 15 of state/build/C75_PHASE0_MAPPING.md: there is no
whole-cycle deterministic-replay test. Read CONTEXT.md (rules 1-29).
You OWN: NEW tests/test_whole_cycle_replay.py only (+ a tiny helper module under tests/ if needed). Do not edit engine code; if replay
is NOT deterministic, find the exact source (unseeded RNG, dict/set iteration order, wall-clock, file-system order, parallel
completion order) and REPORT it with a failing test and the file:line - the main session assigns the fix. Never run git. 1 process.
Do: on the W02 planted world (small config, <= 90 s), run the research loop N cycles twice from scratch with the same seed and assert
the full cycle reports, knowledge, lineage and checkpoints are byte-identical (after removing fields that are legitimately wall-clock,
which you must list and justify); then kill after cycle k, resume from the checkpoint, and assert the resumed run equals the
uninterrupted one; then change the seed and assert something differs (the test can fail). Include the inline and thread executor modes.
Report results and every nondeterminism source found.
