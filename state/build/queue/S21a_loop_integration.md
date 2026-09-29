# S21a_loop_integration (make the learning loop use EVERY module; close the research loop)
Source of truth: state/build/INTEGRATION_AUDIT.md (29 Sep) - 39 queued hooks have no production call site, 17 engine/learning
modules are reachable only from tests. Contract sections 4, 60, 61, 83, 85; canon C58, C61, C64. C63: code + tests; no real-data runs.
Read state/build/CONTEXT.md (rules 1-19), INTEGRATION_AUDIT.md in full, state/build/INTEGRATION.md, SELF_LEARNING_CONTRACT.md §4.
You OWN: engine/learning/learner.py, engine/learning/test_path.py, NEW engine/learning/loop_hooks.py, NEW tests/test_learning_reachability_loop.py.
You may make small API additions (never behaviour changes) in: engine/learning/firewalls.py, knowledge_graph.py (add public
contradiction_keys(now)), research_priority.py, decision_contract.py. S21b owns wiring.py, adaptive.py, improve.py, registry.py,
missed_winners.py, lessons.py, memory.py, patterns.py, scripts/, CI - do not touch those.

Make the section-4 loop genuinely use every learning module on the PRODUCTION path (learner.py <- test_path.py <- livesim_loop2
--learner legit), each where it belongs, through engine/learning/loop_hooks.py:
- decision contract: policy_check + readiness at decide/promotion; combined_influence (calibration) at the contract.
- firewalls consume KnowledgeStore.as_of + audit_future; knowledge.with_failure / with_relation from failure/credit.
- interpretation.next_test + unknowns.rank_unknowns feed the research policy; competition (boundary_field), questions
  (QuestionEngine.ask_many), complexity, hierarchy (shrinkage before retrieval), redundancy (credit.masked_pairs ->
  RedundancyReport.edges -> graph), credit.update_proposals -> BeliefLedger, credit.credit_edges -> graph.
- S15: break_detection, reliability, lifecycle (apply_to_ledger -> retirement), health (inputs_from_knowledge, epistemic_proposals)
  each period; temporal.to_evidence; S20 ContradictionMonitor.run_period + signals into ResearchPriorityEngine; dashboard rows.
- Close the research loop: ResearchPriorityEngine.step -> propose_selected -> (experiment) -> update_from_result;
  signals_from_health and signals_from_data_audit; failed_learners.seed_registry + check_proposal before any new learner.
- disagreement (model disagreement per decision), portfolio_value + scorecard_for_learner -> ScorecardStore.append per period,
  same_year/controls available as the harness entry (a function the runner can call), compute.job_for + checkpoints
  (resume_verified, write_interruption) around test_path runs.
- Persistence: everything learned is written under a configured root (state/learning/...) so it survives the process.
Replace text-presence checking with REACHABILITY: tests/test_learning_reachability_loop.py builds the AST call graph from the
production entry (scripts/livesim_loop2.py main with --learner legit -> test_path -> learner) and asserts every engine/learning
module (except explicitly listed research-only ones, each with a written reason) is reached, and every INTEGRATION.md hook in your
scope has a reachable call site. Plus behavioural tests on the planted world proving data actually flows (e.g. a planted
contradiction reaches the research queue and a result updates it; a credit result changes a belief; a break changes lifecycle).
Keep the learner deterministic; keep every firewall. Report ruler counts (scripts/contract_lines.py) and test results.
