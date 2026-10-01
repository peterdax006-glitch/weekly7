# THE CREATOR — architecture (C77)

Canon C77 (`CREATOR_MASTER_PROMPT.md`, verbatim, hash-locked) is the governing checkoff list. This file is the architect's design
(C77 sec 3: Claude is architect/foundation builder now, evaluator/auditor later). It is a design, not evidence: every capability
below is NOT_STARTED until its work package reaches its states (C77 sec 7) with evidence.

## 1. What the Creator is

An autonomous self-development engine. Given an objective (first: itself, C77 sec 2 and 49) it runs the C77 sec 65 loop:

```text
objective -> requirements -> inspect self -> gaps -> classify -> research -> design options -> strategy -> decompose -> plan
-> allocate -> implement (in a sandbox) -> build -> tests -> evaluate -> failures -> root cause -> repair -> retest -> regression
-> measure improvement vs baseline -> ADOPT or ROLLBACK -> record -> evaluate strategy -> improve strategy -> objective met? -> repeat
```

and the sec 66 recursive loop over its own development process.

## 2. Design principles (each traceable to C77)

| Principle | C77 |
|---|---|
| Every state transition is an append-only, hash-chained ledger record with provenance; nothing mutates silently | 35, 37, 58, 61 |
| The self-model is DERIVED from evidence (AST, imports, pytest collection/results, git, checklists, ruler, reachability); a claim without evidence is UNTESTED | 11, 41, 42 |
| All code changes happen in an isolated git worktree sandbox; adoption is a controlled merge; rollback is discarding/reverting | 22, 44, 45 |
| A change is an IMPROVEMENT only with baseline + controlled comparison + measurable benefit + regression check + reproducibility + holdout | 30, 31, 46 |
| Development is measured on a SEALED development-task suite (dev split for tuning, holdout split sealed and never used to tune) | 32, 55, 77 |
| The kernel can never edit protected paths (canon, contracts, sealed suites, evaluator, auditor, firewall config) | 33, 34, 44 |
| LLM agents (Claude Code CLI, headless `claude -p`) are WORKERS with roles; the orchestration, evidence and decisions are the kernel's | 21, 57 |
| No task-specific code paths: the auditor scans the kernel for benchmark/task identifiers and hardcoded demonstrations | 50, 56, 68 |
| Uncertainty is explicit: KNOWN / LIKELY / UNCERTAIN / UNKNOWN / CONTRADICTED / UNTESTED / FAILED / VALIDATED | 71 |
| Weekly7 is FOUNDATION: reuse its infrastructure (provenance, checkpoints, experiment memory, sequential evidence, ruler, checklist tooling, reachability, replay tests), extend, never duplicate | 5, 82 |

## 3. Components (package `creator/`)

| # | Component | Responsibility | Reuses from the foundation |
|---|---|---|---|
| K01 | `ledger` | append-only hash-chained development ledger; typed records (Objective, Requirement, Gap, Question, Finding, DesignOption, WorkPackage, ChangeProposal, Experiment, TestRun, Failure, Diagnosis, Repair, Measurement, ImprovementClaim, Decision, StrategyOutcome); provenance on every record | `engine/provenance.py`, archive chain files |
| K02 | `selfmodel` | evidence scanners -> machine-readable self-model: components, interfaces, dependency graph, tests (collected + last results), capabilities with evidence + uncertainty state, limitations, resources, versions, active work; diff against claims (self-diagnosis) | `scripts/reachability.py`, `scripts/contract_lines.py`, checklists, git |
| K03 | `objective` | objective -> structured problem (outcomes, constraints, non-goals, acceptance criteria, metrics, risks, unknowns); requirement compiler (id, priority, dependency, acceptance test, measurement, failure condition, evidence location, state); goal-drift check | — |
| K04 | `gaps` | requirements vs self-model -> gaps classified CAPABILITY / KNOWLEDGE / ARCHITECTURE / TESTING / INTEGRATION / EVIDENCE, with dependencies and importance | — |
| K05 | `agents` | role-based worker runtime over the Claude Code CLI (researcher, architect, implementer, tester, debugger, adversary, validator, auditor): prompts from templates, tool permissions per role, working directory per sandbox, budgets, timeouts, transcript capture as evidence, result schemas validated | — |
| K06 | `sandbox` | git-worktree sandboxes per change; build (import/compile/mypy), test selection (affected tests via import graph) and execution, result capture; adopt (commit + merge) / rollback (discard / revert); protected-path firewall | git, CI commands, `pyproject.toml` mypy |
| K07 | `research` | question generation/prioritisation, evidence collection (repo, docs, web via agents), source evaluation, contradiction detection, confidence, research memory, research->design bridge | `engine/research/questions.py`, `priority.py` ideas |
| K08 | `design` | N alternative designs per decision, assumptions, failure modes, cost/risk estimates, adversarial critique, explicit selection criteria, rejected alternatives kept | — |
| K09 | `planner` | decomposition into C77 sec 63 work packages, dependency DAG, critical path, parallelism, resource allocation, dynamic replanning on evidence | `compute_manager`, `value_accounting` ideas |
| K10 | `evaluate` | metric registry, baseline vs candidate measurement, uncertainty (sequential evidence), regression accounting, reproducibility re-runs, holdout gate, IMPROVEMENT / NO_EFFECT / REGRESSION / INSUFFICIENT_EVIDENCE verdicts computed, never typed | `engine/research/evidence.py` SequentialPlan |
| K11 | `debug` | failure reproduction, localisation, classification (C76 taxonomy), hypotheses, evidence, root cause, repair proposals, repair experiments | — |
| K12 | `memory` | development history, failure / repair / research / architecture / strategy / resource memory; "seen this before?" retrieval; never a hidden answer key | `engine/learning/experiment_memory.py` |
| K13 | `meta` | development STRATEGIES as explicit parameterised policies (research budget, design breadth, test-first vs code-first, reviewer depth, agent choice, retry rules); outcome tracking per problem class; strategy selection; process-improvement proposals that target the kernel's own policies and code (sec 39-40, 53-54) | `failed_lab`, `meta_research` ideas |
| K14 | `kernel` | the sec 65 loop orchestrator: state machine, checkpoints (resume after crash, never assume an interrupted task succeeded), observability snapshot (current objective, plan, task, hypothesis, versions, experiments, results, changes, rollbacks, resources, confidence, open questions) | loop checkpoint ideas |
| K15 | `devbench` | the SEALED development-task suite that measures the Creator: self-contained tasks in a scratch repo (planted bugs with hidden tests, missing features with hidden acceptance tests, refactors with invariants, test-gap tasks, performance tasks), answer keys sealed and hashed BEFORE runs (C72 pattern), dev / holdout split; scores success, cost, regressions, false "done" claims | `pattern_benchmark` sealed-key pattern |
| K16 | `audit` | independent auditor + adversary: separate read-only agent sessions with the right to return INSUFFICIENT_EVIDENCE; computed checks for hardcoding, holdout contamination, provenance gaps, fake autonomy (were intermediate decisions made by the kernel or supplied by Claude?) | — |

## 4. Measuring the Creator without faking it

- **Level 1 (it can develop):** the share of `devbench` tasks solved to the hidden acceptance tests, the false-completion rate, the regression rate, and the cost.
- **Level 2 (development improved):** a strategy or process change is adopted only if `devbench` dev scores improve under the evaluate
  gate AND the sealed holdout does not regress.
- **Level 3 (the process of improving improved):** the same measurement applied to changes in the meta layer, comparing development
  outcomes before and after across different task categories (sec 51-55).
- **Self-development objective (sec 49):** the kernel is pointed at `creator/` itself with only "develop yourself" plus the standing
  constraints. Gaps, designs and fixes come from its own machinery. The audit checks that no step was supplied by Claude.

## 5. Safety and integrity

- Protected paths: `canon/`, contract texts, `*.lock.json`, `creator/devbench/sealed/`, `creator/audit/`, the evaluator's thresholds
  file, and CI config. Changes there are refused at the sandbox layer and flagged by the auditor.
- Resource limits: agent-call budget per objective, wall-clock limits, the RAM rule (>= 2.5 GB free), at most N concurrent sandboxes.
- Weekly7 C64/C56 boundaries are untouched; the Creator never reads `state/livesim/` or sealed answer keys.

## 6. Build order (work packages; each gets a C77 sec 63 package file)

1. **Ledger and data model:** CR01.
2. **Self-model:** CR02.
3. **Sandbox, build, test and rollback:** CR03.
4. **Agent runtime:** CR04.
5. **`devbench` with sealed keys:** CR05.
6. **Evaluate:** CR06.
7. **Objective and requirements:** CR07.
8. **Gaps:** CR08.
9. **Planner:** CR09.
10. **Research:** CR10.
11. **Design:** CR11.
12. **Debug:** CR12.
13. **Memory:** CR13.
14. **Kernel loop with checkpoints and observability:** CR14.
15. **Meta / strategies:** CR15.
16. **Audit and adversary:** CR16.
17. **First self-development test:** CR17.
18. **Second and third tests:** CR18.
19. **Recursive test:** CR19.
20. **Final adversarial campaign:** CR20.

Foundation repair on Weekly7 is done only where the Creator needs it to operate (C77 sec 4).

## 7. Meaningful-depth floors (C77 sec 62: set per subsystem from required complexity; re-assessed by the auditor)

| Component | Floor (meaningful lines, ruler) | Component | Floor |
|---|---:|---|---:|
| K01 ledger | 900 | K09 planner | 1,200 |
| K02 selfmodel | 1,800 | K10 evaluate | 1,400 |
| K03 objective | 1,000 | K11 debug | 1,200 |
| K04 gaps | 700 | K12 memory | 900 |
| K05 agents | 1,400 | K13 meta | 1,500 |
| K06 sandbox | 1,400 | K14 kernel | 1,600 |
| K07 research | 1,200 | K15 devbench | 1,800 + tasks |
| K08 design | 900 | K16 audit | 1,200 |

These floors are guards against shallowness, not targets. Tests are counted separately and are required for every component.
