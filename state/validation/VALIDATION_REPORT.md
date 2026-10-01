# Creator independent validation report

| K | module | verdict | tests | issues |
|---|---|---|---|---|
| K01 | ledger | VALIDATED | 21 passed in 24.49s | 0 |
| K02 | selfmodel | VALIDATED | 9 passed in 90.62s (0:01:30) | 0 |
| K03 | objective | VALIDATED_WITH_ISSUES | 5 passed in 35.87s | 1 |
| K04 | gaps | VALIDATED_WITH_ISSUES | 8 passed in 359.59s (0:05:59) | 1 |
| K05 | agents | VALIDATED_WITH_ISSUES | 13 passed in 8.36s | 1 |
| K06 | sandbox | VALIDATED | 32 passed in 29.31s | 0 |
| K07 | research | VALIDATED | 10 passed in 0.99s | 0 |
| K08 | design | VALIDATED | 7 passed in 0.16s | 0 |
| K09 | planner | VALIDATED | 7 passed in 48.66s | 0 |
| K10 | evaluate | VALIDATED_WITH_ISSUES | 8 passed in 115.40s | 1 |
| K11 | debug | VALIDATED_WITH_ISSUES | 11 passed in 5.82s | 2 |
| K12 | memory | VALIDATED | 6 passed in 1.56s | 0 |
| K13 | meta | VALIDATED_WITH_ISSUES | 7 passed in 0.15s | 1 |
| K14 | kernel | VALIDATED_WITH_ISSUES | 21 passed in 331.65s before; 22 passed in 403.28s after round-3 fixes | 3 |
| K15 | devbench | VALIDATED | 10 passed in 171.45s (0:02:51) | 0 |
| K16 | audit | VALIDATED | 19 passed in 114.86s | 0 |
| K17 | efficiency | VALIDATED | 11 passed in 131.66s | 0 |
| K18 | generator | VALIDATED_WITH_ISSUES | 22 passed in 353.55s (0:05:53) | 4 |
| K19 | autotune | VALIDATED | 11 passed in 5.21s | 0 |
| K20 | localworker | VALIDATED_WITH_ISSUES | 3 passed in 24.06s | 1 |
| K21 | selfworkers | VALIDATED_WITH_ISSUES | 8 passed in 40.51s | 1 |
| K22 | swarm | VALIDATED_WITH_ISSUES | 8 passed in 185.31s before; 9 passed in 189.86s after round-3 fixes | 3 |
| K23 | oversight | VALIDATED_WITH_ISSUES | 7 passed in 79.06s | 1 |
| K24 | recursion | VALIDATED_WITH_ISSUES | 5 passed in 1.58s | 1 |
| K25 | testgen | VALIDATED | 21 passed in 86.29s (0:01:26) | 0 |
| K26 | reproduce | VALIDATED | 11 passed in 55.27s | 0 |
| NEW-curriculum | curriculum | VALIDATED_WITH_ISSUES | 9 passed before; 10 passed after fixes | 5 |
| NEW-student | student | VALIDATED_WITH_ISSUES | 3 passed before; 4 passed after fix | 4 |
| NEW-process_levers | process_levers | VALIDATED_WITH_ISSUES | 6 passed before; 8 passed after fixes | 3 |
| NEW-synth | synth | VALIDATED_WITH_ISSUES | 7 passed in 0.28s | 3 |

Counts: {"VALIDATED": 13, "VALIDATED_WITH_ISSUES": 17}

## Issues
- K03 [medium] creator/objective.py:170: compile_capabilities ordered by id, silently dropping the K10->K15 build-order dependency (K15 not yet compiled). FIXED, regression test.
- K03 [low] creator/objective.py:52: COMPONENT_DEPENDS covers only K01-K16; K17-K26 have no build-order dependencies. FIXED (h3/validator2).
- K03 [low] creator/objective.py:148: register_objective is idempotent on statement only: a changed outcomes/constraints list is silently ignored.
- K04 [low] creator/gaps.py:1: Test suite takes ~6 minutes (needs real selfmodel build); no cheap unit-level coverage of sync() regression/stale branches observed.
- K05 [medium] creator/agents.py:218: run_agent crashed (AttributeError) when the CLI printed valid non-object JSON ('"x"', 123, null). FIXED, regression test.
- K05 [low] creator/agents.py:5: Docstring says no user/project settings are loaded, but build_command passes --setting-sources project. FIXED (h3/validator2).
- K05 [low] creator/agents.py:98: BudgetError text tells the operator to set CREATOR_AGENT_CALLS=1, which from_env deliberately ignores: misleading. FIXED (h3/validator2).
- K05 [low] creator/agents.py:118: Budget.record is an unlocked read-modify-write; two concurrent swarm workers can lose a call record (budget can be undercounted).
- K07 [low] creator/research.py:43: generate_questions: float(importance) raises on non-numeric/None importance instead of skipping/defaulting; docstring promises only description-less skipping. FIXED (h3/validator2).
- K08 [low] creator/design.py:118: select() keys scores by option name: duplicate names collapse (only generate_options dedupes), so a duplicate can be scored as its twin. FIXED (h3/validator2).
- K10 [low] creator/evaluate.py:190: Both guard Measurements of one arm cite the same pooled evidence file path (rewritten per guard); content identical today but fragile if guards ever diverge.
- K11 [low] creator/debug.py:87: reproduce(): on timeout stderr is dropped from the captured output.
- K11 [low] creator/debug.py:24: ASSERTION regex 'assert ' matches any text containing the word; fallback classification can mislabel.
- K12 [low] creator/memory.py:108: load() renumbers seq after skipping corrupt lines, so seq differs from what was saved; save() is not atomic. FIXED (h3/validator2).
- K13 [low] creator/meta.py:132: record() accepts NaN cost; Strategy.agent is not checked against any agent registry.
- K19 [medium] creator/autotune.py:63: tried() json-decodes every history line unguarded: one corrupt line makes propose()/trial() raise forever. FIXED (h3/validator2).
- K20 [low] creator/localworker.py:42: relevant_files(package, workdir): workdir is an unused (dead) parameter.
- K20 [medium] creator/localworker.py:54: run_tests: subprocess.run timeout raises TimeoutExpired, uncaught; it escapes LocalWorker.__call__ instead of being a failed attempt. FIXED (h3/validator2).
- K21 [low] creator/selfworkers.py:135: dead_private/_used_names skip= only skips the node itself, not its children: a self-recursive unused private is never deleted (safe direction, but docstring over-promises).
- K23 [medium] creator/oversight.py:58: goal_drift docstring promises flagging concentrated work 'while the goal's metrics do not move'; code never checks metrics and 'concentrated' does not set drifting. FIXED (h3/validator2).
- K23 [low] creator/oversight.py:84: knowledge_gaps: ledger.get(failure_id) raises if a Diagnosis cites a missing failure.
- K24 [low] creator/recursion.py:1: Docstring labels itself K20 (declared as K24). FIXED (h3/validator2).
- K24 [low] creator/recursion.py:235: step(forced=Change) is not validated against meta.BOUNDS or PARAMS.

- K14 [medium] creator/kernel.py:580: A worker handing a package over (curriculum handed-over kind) returned WorkResult(False) and was treated as 'worker changed nothing' = a failed attempt; planner.MAX_ATTEMPTS=3 then BLOCKED the gap, so the handed-over task was silently dropped (deferred.jsonl is written but nothing reads it). FIXED: WorkResult.deferred, outcome DEFERRED, planner.attempts_for ignores deferred packages; regression test_a_deferred_package_is_not_a_failed_attempt_and_is_never_blocked_away (failed before).
- K14 [low] creator/kernel.py:703: kernel_log.jsonl is appended by every swarm thread with an unlocked text write; a very large CycleReport (> write buffer) could interleave. Not fixed.
- K14 [info] creator/kernel.py:621: Core invariant CONFIRMED: adoption requires verdict IMPROVEMENT and ev.clean and no weakening/planted answers (any reason raises _Reject); WorkResult.claimed_done is never consulted, so a worker's self-report cannot adopt.
- K18 [medium] creator/generator.py:1: Docstring: 'holdout tasks are never recorded'. Nothing enforces it: the solver is never told the split (Task.to_public hides it) and devbench.score_holdout does not force learn=False; only caller discipline (test_holdout_runs_never_write_experience passes learn=False by hand). devbench.py is out of scope to edit. Not fixed.
- K18 [low] creator/generator.py:561: solve_with_search leaves the last mutated candidate on disk if visible_tests raises (no try/finally); sandbox is discarded afterwards so harmless in the kernel.
- K18 [low] creator/generator.py:243: ExperienceMemory.add is an unlocked text append (files in each row): unsafe if two threads ever share one memory. The swarm filler uses SearchSolver without memory, so not reachable today.
- K18 [info] creator/generator.py:1: 'model' strategy needs llama-server + weights; its tests use a FakeLLM, the real path is untested here.
- K22 [medium] creator/swarm.py:196: An exception in a job thread (e.g. make_worker or Sandbox.open failing) left no CycleReport and the package's ledger chain IN_PROGRESS: the package vanished from the round. FIXED: job catches, records FAILED, reports ERROR; regression test_a_worker_that_crashes_before_the_kernel_runs_is_reported_not_lost (failed before).
- K22 [low] scripts/creator_swarm.py:109: on_report lambda tuple trick tripped mypy func-returns-value. FIXED (named function).
- K22 [info] creator/swarm.py:115: Races checked: ledger appends use the Ledger file lock; repo-mutating steps share one lock; WAITING is locked, HANDED_BACK is plain set add/read (GIL-atomic). The lessons.jsonl / deferred.jsonl race was in curriculum.py (see NEW-curriculum).
- NEW-curriculum [high] creator/curriculum.py:76: LessonLog appended with unlocked text writes; a lesson carries whole files (far above the write buffer) and 8 swarm threads share lessons.jsonl and deferred.jsonl: lines interleaved and lessons() silently dropped the corrupt ones. FIXED: module lock + single binary write; regression test_concurrent_appends_never_interleave_or_lose_lessons (failed before).
- NEW-curriculum [high] creator/curriculum.py:~253: Docstring: a handed-over task is 'queued in deferred.jsonl and retried by the kernel later'. Nothing reads deferred.jsonl and the kernel counted the deferral as a failed attempt, so after 3 rounds the gap was BLOCKED (dropped). FIXED via WorkResult.deferred (see K14); deferred.jsonl remains an audit log only.
- NEW-curriculum [info] creator/curriculum.py:236: Self-report cannot count as adoption: Lesson.adopted is set only by resolve() from the kernel CycleReport (ADOPTED -> True, REJECTED/ROLLED_BACK -> False, anything else None = not scored). teacher_share/claude_share count only adopted lessons, i.e. measured outcomes. CONFIRMED.
- NEW-curriculum [low] creator/curriculum.py:~210: A lesson whose cycle ends CANCELLED/ERROR stays adopted=None forever (not scored either way); a student that returns not-done is scored a failure immediately.
- NEW-curriculum [low] creator/curriculum.py:~268: A student that owns a kind but cannot_attempt/fails sends the package to deferral, not to Claude, by design; if the student never recovers the kind still cannot progress (only attempts of the student can lower its rate).
- NEW-student [high] creator/student.py:~215: reload() read lessons.jsonl line by line and required adopted=True ON the lesson line, but curriculum.LessonLog writes lessons with adopted=None and the kernel's verdict as a separate {'event':'outcome'} line: against a real lessons.jsonl the student learned nothing (its tests used a hand-made flat format: vacuous w.r.t. the real writer). FIXED: outcome events are joined; regression test_learns_from_the_real_lesson_log_format (failed before).
- NEW-student [low] creator/student.py:~330: __call__ reads package.task_kind which WorkPackage lacks (always ''); template ranking by task kind never applies when invoked through the kernel. can_attempt(lesson) does use the kind.
- NEW-student [low] creator/student.py:~335: write_text on Windows rewrites a LF file with CRLF (whole-file diff); the kernel measures it either way.
- NEW-student [info] creator/student.py:1: The student never judges itself: it returns claimed_done only; the kernel measures.
- NEW-process_levers [high] creator/process_levers.py:59: ProcessSolver passed per_family and pair_width POSITIONALLY into the wrong parameters of solve_with_search (per_family landed in `rank`, pair_width in `extra_passes`): design_breadth and reviewer_depth never reached the search, contradicting the module docstring (mypy arg-type caught it). FIXED with keywords; regression test_process_solver_passes_the_levers_to_the_right_search_parameters (failed before).
- NEW-process_levers [medium] creator/process_levers.py:77: load_process accepted any integer from process.json (max_retries=10**9, research_budget=-5). FIXED: values outside meta.BOUNDS fall back to the default process; regression test_load_process_rejects_out_of_bounds_values (failed before).
- NEW-process_levers [low] creator/process_levers.py:119: DevWorkload with fewer dev tasks than chunk*reps+confirm yields empty chunks scored 0.0/0.0 instead of an error.
- NEW-synth [medium] creator/synth.py:218: Idiom bodies and enumerated expressions (pow, range-driven loops) are exec'd/eval'd on example values with no time or size limit: a large example (fibonacci(10**9), pow(10, 10**9)) hangs the solver. Not fixed (devbench task timeouts bound it).
- NEW-synth [low] creator/synth.py:30: Idioms are docstring-vocabulary keyed per benchmark-style task; docstring says function names/ids/answers are never consulted and the code agrees (only the spec's own examples are used). Hidden tests still judge; no leak found.
- NEW-synth [low] creator/synth.py:300: solve_file rewrites the whole file via ast.unparse (comments lost); writes only when every stub is solved.

## Round 3 (h4/validator3)
- validator3 (h4/validator3): K14, K18, K22 validated; NEW-curriculum/student/process_levers/synth added. Fixed with regression tests failing first: deferred package dropped by MAX_ATTEMPTS, lessons.jsonl concurrent-append corruption, student never learned from the real lesson log, process levers mis-wired into search, unbounded process.json, swarm job crash lost its report. mypy: creator/debug.py has 6 pre-existing errors (not touched).

## Fixed in round 2 (h3/validator2; each regression test failed before its fix)
- goal_drift metric stall, run_tests timeout, corrupt autotune history, BudgetError text, agents docstring, select duplicate names, research importance, memory seq, recursion docstring, COMPONENT_DEPENDS K17-K26
- Round 2 test results: oversight 7 passed, localworker 3, efficiency 11, evaluate 8, planner 7, selfworkers 8, sandbox 32, new regressions 8. K14, K18, K22 still for the orchestrator.

## Fixed (with regression tests proven failing first)
- agents.run_agent crashed on non-object JSON CLI output
- objective.compile_capabilities dropped the K10->K15 dependency

Depth note: components marked partial were validated mainly by running their declared tests plus targeted reading, not a full line-by-line audit.