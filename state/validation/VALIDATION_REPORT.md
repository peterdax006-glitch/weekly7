# Creator independent validation report

| K | module | verdict | tests | issues |
|---|---|---|---|---|
| K01 | ledger | VALIDATED | 21 passed in 24.49s | 0 |
| K02 | selfmodel | VALIDATED | 9 passed in 90.62s (0:01:30) | 0 |
| K03 | objective | VALIDATED_WITH_ISSUES | 5 passed in 35.87s | 3 |
| K04 | gaps | VALIDATED_WITH_ISSUES | 8 passed in 359.59s (0:05:59) | 1 |
| K05 | agents | VALIDATED_WITH_ISSUES | 13 passed in 8.36s | 4 |
| K06 | sandbox | NOT_VALIDATED | STILL RUNNING at report time (no failure seen so far):  | 0 |
| K07 | research | VALIDATED_WITH_ISSUES | 10 passed in 0.99s | 1 |
| K08 | design | VALIDATED_WITH_ISSUES | 7 passed in 0.16s | 1 |
| K09 | planner | NOT_VALIDATED | STILL RUNNING at report time (no failure seen so far):  | 0 |
| K10 | evaluate | NOT_VALIDATED | STILL RUNNING at report time (no failure seen so far):  | 1 |
| K11 | debug | VALIDATED_WITH_ISSUES | 11 passed in 5.82s | 2 |
| K12 | memory | VALIDATED_WITH_ISSUES | 6 passed in 1.56s | 1 |
| K13 | meta | VALIDATED_WITH_ISSUES | 7 passed in 0.15s | 1 |
| K14 | kernel | NOT_VALIDATED | STILL RUNNING at report time (no failure seen so far):  | 0 |
| K15 | devbench | VALIDATED | 10 passed in 171.45s (0:02:51) | 0 |
| K16 | audit | VALIDATED | 19 passed in 114.86s | 0 |
| K17 | efficiency | NOT_VALIDATED | STILL RUNNING at report time (no failure seen so far):  | 0 |
| K18 | generator | NOT_VALIDATED | STILL RUNNING at report time (no failure seen so far):  | 0 |
| K19 | autotune | VALIDATED_WITH_ISSUES | 11 passed in 5.21s | 1 |
| K20 | localworker | NOT_VALIDATED | STILL RUNNING at report time (no failure seen so far):  | 2 |
| K21 | selfworkers | NOT_VALIDATED | STILL RUNNING at report time (no failure seen so far):  | 1 |
| K22 | swarm | NOT_VALIDATED | STILL RUNNING at report time (no failure seen so far):  | 0 |
| K23 | oversight | NOT_VALIDATED | STILL RUNNING at report time (no failure seen so far):  | 2 |
| K24 | recursion | VALIDATED_WITH_ISSUES | 5 passed in 1.58s | 2 |
| K25 | testgen | VALIDATED | 21 passed in 86.29s (0:01:26) | 0 |
| K26 | reproduce | VALIDATED | 11 passed in 55.27s | 0 |

Counts: {"VALIDATED": 6, "VALIDATED_WITH_ISSUES": 10, "NOT_VALIDATED": 10}

## Issues
- K03 [medium] creator/objective.py:170: compile_capabilities ordered by id, silently dropping the K10->K15 build-order dependency (K15 not yet compiled). FIXED, regression test.
- K03 [low] creator/objective.py:52: COMPONENT_DEPENDS covers only K01-K16; K17-K26 have no build-order dependencies.
- K03 [low] creator/objective.py:148: register_objective is idempotent on statement only: a changed outcomes/constraints list is silently ignored.
- K04 [low] creator/gaps.py:1: Test suite takes ~6 minutes (needs real selfmodel build); no cheap unit-level coverage of sync() regression/stale branches observed.
- K05 [medium] creator/agents.py:218: run_agent crashed (AttributeError) when the CLI printed valid non-object JSON ('"x"', 123, null). FIXED, regression test.
- K05 [low] creator/agents.py:5: Docstring says no user/project settings are loaded, but build_command passes --setting-sources project.
- K05 [low] creator/agents.py:98: BudgetError text tells the operator to set CREATOR_AGENT_CALLS=1, which from_env deliberately ignores: misleading.
- K05 [low] creator/agents.py:118: Budget.record is an unlocked read-modify-write; two concurrent swarm workers can lose a call record (budget can be undercounted).
- K07 [low] creator/research.py:43: generate_questions: float(importance) raises on non-numeric/None importance instead of skipping/defaulting; docstring promises only description-less skipping.
- K08 [low] creator/design.py:118: select() keys scores by option name: duplicate names collapse (only generate_options dedupes), so a duplicate can be scored as its twin.
- K10 [low] creator/evaluate.py:190: Both guard Measurements of one arm cite the same pooled evidence file path (rewritten per guard); content identical today but fragile if guards ever diverge.
- K11 [low] creator/debug.py:87: reproduce(): on timeout stderr is dropped from the captured output.
- K11 [low] creator/debug.py:24: ASSERTION regex 'assert ' matches any text containing the word; fallback classification can mislabel.
- K12 [low] creator/memory.py:108: load() renumbers seq after skipping corrupt lines, so seq differs from what was saved; save() is not atomic.
- K13 [low] creator/meta.py:132: record() accepts NaN cost; Strategy.agent is not checked against any agent registry.
- K19 [medium] creator/autotune.py:63: tried() json-decodes every history line unguarded: one corrupt line makes propose()/trial() raise forever.
- K20 [low] creator/localworker.py:42: relevant_files(package, workdir): workdir is an unused (dead) parameter.
- K20 [medium] creator/localworker.py:54: run_tests: subprocess.run timeout raises TimeoutExpired, uncaught; it escapes LocalWorker.__call__ instead of being a failed attempt.
- K21 [low] creator/selfworkers.py:135: dead_private/_used_names skip= only skips the node itself, not its children: a self-recursive unused private is never deleted (safe direction, but docstring over-promises).
- K23 [medium] creator/oversight.py:58: goal_drift docstring promises flagging concentrated work 'while the goal's metrics do not move'; code never checks metrics and 'concentrated' does not set drifting.
- K23 [low] creator/oversight.py:84: knowledge_gaps: ledger.get(failure_id) raises if a Diagnosis cites a missing failure.
- K24 [low] creator/recursion.py:1: Docstring labels itself K20 (declared as K24).
- K24 [low] creator/recursion.py:235: step(forced=Change) is not validated against meta.BOUNDS or PARAMS.

## Fixed (with regression tests proven failing first)
- agents.run_agent crashed on non-object JSON CLI output
- objective.compile_capabilities dropped the K10->K15 dependency

Depth note: components marked partial were validated mainly by running their declared tests plus targeted reading, not a full line-by-line audit.