# CR03 - Sandbox, build, test, adopt, rollback  [model: opus]
PACKAGE ID: CR03 (C77 sec 63 format). Governing text: CREATOR_MASTER_PROMPT.md (C77) sections 22-25, 28, 29, 44-46, 59-61;
creator/ARCHITECTURE.md (K06, sec 5 safety). CONTEXT.md rules 1-29 apply (NEVER git stash / reset / rebase in the main tree).
OBJECTIVE: the controlled environment in which every autonomous code change is made, built, tested, compared and adopted or rolled back.
WHY IT EXISTS: C77 sec 44 - an unverified self-change must never silently become authoritative; sec 45 - every change needs a recovery path.
PREREQUISITES: none hard; integrate with CR01's ledger interface when it exists (write records through a small adapter so CR03 can be built
in parallel; specify the adapter's contract). INPUTS: the weekly7 git repo, CI commands (.github/workflows/ci.yml), pyproject mypy config.
IMPLEMENTATION REQUIREMENTS (creator/sandbox.py, creator/build.py, creator/testrun.py):
- Sandbox = `git worktree add` on a new branch from a recorded base commit, in a scratch directory OUTSIDE the main tree; cleanup on
  close; never touches the main working tree's index or files; refuses to start if the base commit is unknown.
- Protected-path firewall: a change touching canon/, *_PROMPT.md / contract texts, *.lock.json, creator/devbench/sealed/, creator/audit/,
  .github/ or the evaluator thresholds file is REFUSED (fail closed) with a record.
- Build: import every changed module, compile, run CI's mypy command in the sandbox; classify build failures (syntax, import, type,
  environment) with the raw output kept as evidence.
- Tests: select affected tests via the import graph (changed module -> tests importing it, transitively) plus an always-run smoke set;
  run with pytest in the sandbox; parse results (junit xml) into TestRun records; detect NEW failures vs the baseline run on the base
  commit (regression = failing now, passing at base); timeouts; flaky-test detection by re-run.
- Adopt: only through an explicit decision object (from the evaluator, later) -> commit in the sandbox branch, then merge into main with
  `git merge --no-ff` from the main tree ONLY if the main tree is clean of conflicts with the change; never force; record the merge.
  Rollback: discard the worktree; for an already-adopted change, `git revert` of its merge commit. Both recorded.
INTERFACES: Sandbox.open(base) / apply(patch or files) / build() / test(selection) / diff() / close(); adopt(decision); rollback(change_id).
TEST REQUIREMENTS (tests/test_creator_sandbox.py, < 90 s, using a TEMPORARY git repo fixture, never the real repo): open/close leaves
main untouched; protected path refused; build failure classified; affected-test selection; regression detection vs base; flaky re-run;
adopt merges; rollback discards and reverts; interrupted sandbox recovered (stale worktree pruned) and never assumed successful.
EXPECTED FAILURE MODES: worktree left behind, main index touched, merge with conflicts forced, a test failing at base counted as regression.
EVIDENCE: test output + creator/packages/CR03_REPORT.md. ROLLBACK: additive package. COMPLETION: implemented + tested; mypy clean;
ruler >= 1,400 meaningful lines (K06) without padding. ANTI-PREMATURE: never VALIDATED here. STATUS: NOT_STARTED. Never run git on the
MAIN repo yourself (tests use temp repos); <= 1 process.
