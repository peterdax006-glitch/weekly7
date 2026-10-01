# CR00_phase0_reconnaissance  [model: opus]  (READ-ONLY: write only the two output files)
Canon C77 = THE CURRENT master checkoff list, to be followed verbatim: C:\Users\Peter\weekly7\CREATOR_MASTER_PROMPT.md. Read it fully
first (sections 0-87). The product is now THE CREATOR - an autonomous self-development engine; Weekly7 / C62-C68 are its FOUNDATION and
test environment. Read state/build/CONTEXT.md rules 1-29 and Masterstock/MASTERSTOCK.md "NEXT SESSION: START HERE".
Do C77 section 6 (Phase 0 - complete reconnaissance) with this focus: what in the existing repo the Creator can REUSE as development
infrastructure, and what it lacks. Inspect (evidence, not checklist notes): repository structure; git state; tests (count, runtime, the
last full-suite result in state/build/evidence/); CI (.github/workflows/ci.yml); provenance (engine/provenance.py, engine_tree_hash,
verify_integrity); rollback (git, checkpoints: engine/learning/checkpoints.py, loop checkpoints); deterministic replay
(tests/test_whole_cycle_replay*.py); reachability (scripts/reachability.py); data flow; firewalls and research-only / learner-truth
boundaries (C64, pattern_benchmark sealed keys); line ruler (scripts/contract_lines.py); checklist machinery
(scripts/contract_checklist.py, apply_corrections.py, *_CHECKLIST.json); experiment memory (engine/learning/experiment_memory.py,
compute.py); research machinery that is GENERIC enough to serve development (questions.py, priority.py, hypothesis_tree.py,
compute_manager.py, value_accounting.py, waste.py, failed_lab.py, meta_learning / meta_research, replication.py, quality_gate.py,
evidence.py SequentialPlan, scorecard, brain_health, diversity, science_memory, research_graph); health/monitoring; compute/resource
controls (RAM rules); the Claude Code CLI (`claude` 2.1.286 is installed - headless `claude -p` can serve as an implementer/researcher/
reviewer the Creator orchestrates); known failures and limitations (C75_PHASE0_MAPPING.md, C75_PHASE0_REPORT.md, journal); stale
checklist states; duplicates; disconnected or shallow implementations; future-leak risks.
Write:
1. state/build/CREATOR_PHASE0_LEDGER.md - the authoritative current-state ledger. Every finding classified (C77 sec 6) as known
   capability / known limitation / unresolved question / work item / dependency / risk / validated evidence / failed evidence /
   scientific limitation, with file:line or artifact evidence. A section "REUSABLE FOR THE CREATOR" mapping each C77 kernel stage
   (sec 10: objective, self-model, requirements, capability map, knowledge-gap, research, design, decomposition, planning, resources,
   implementation, build, test, evaluation, failure analysis, repair, regression, improvement measurement, memory, meta-learning,
   self-improvement) to existing code that could serve it (with its honest state) or "nothing exists".
2. state/build/CREATOR_PHASE0_LEDGER.json - the same findings machine-readable (id, class, title, evidence, related C77 section).
Never edit code/checklists/tests; git read-only; <= 1 process; never kill processes. Report: counts per class, the reuse map summary,
the top risks for building the Creator, and anything that blocks Phase 1.
