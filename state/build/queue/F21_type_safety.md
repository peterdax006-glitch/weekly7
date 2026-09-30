# F21_type_safety  [model: sonnet]
C75 Phase 0 baseline and Phase 17 "type safety clean"; C62 section 61. Read state/build/CONTEXT.md (rules 1-29) and pyproject.toml [tool.mypy].
Baseline 30 Sep (state/build/evidence/c75_phase0/mypy.txt): the CI command `python -m mypy` (scope engine/learning) FAILS with 3 errors
in 2 files (e.g. engine/learning/retirement.py:810); the override block that sets ignore_errors for engine.learning.learner,
same_year, contradiction_monitor, test_path was meant to be temporary (S17b) and was never removed; engine/research is not type-checked
at all (928 errors across learning+research when both are checked with --ignore-missing-imports).
You OWN: type annotations and type-only fixes in engine/learning/*.py and engine/research/*.py, and pyproject.toml [tool.mypy].
Rules: fix types, never behaviour. A change that alters a runtime result is out of scope - if a type error reveals a REAL bug, write
a failing test, fix it minimally and list it separately. Never add `# type: ignore` or `Any` to silence an error unless the third-party
stub is genuinely missing (say which). Never run git. Other builders (F19 benchmark: engine/research/pattern_benchmark.py; F20
read-only audit) and a real-data Test loop are running: use <= 1 process, and do not touch engine/research/pattern_benchmark.py.
Do, in order, committing nothing (the main session commits):
1. Make `python -m mypy` (the CI command) pass: fix the 3 current errors.
2. Delete the ignore_errors override and fix learner.py, same_year.py, contradiction_monitor.py, test_path.py.
3. Add engine/research to the mypy `files` and fix it module by module (largest error counts first). If the full package cannot be
   finished in this brief, add the finished modules explicitly and list the remaining ones with their error counts - never use a
   package-wide ignore.
4. After each step run the affected module tests; at the end run tests/test_learning_*.py and tests/test_research_*.py.
Report: error counts before/after per module, real bugs found (with tests), test results.
