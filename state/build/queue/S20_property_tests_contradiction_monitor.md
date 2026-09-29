# S20_property_tests_contradiction_monitor
Contract sections 18, 46, 62 (property + determinism tests); checklist I03 (property tests), J09 (contradiction monitoring).
Read state/build/CONTEXT.md fully (rules 1-19) and SELF_LEARNING_CONTRACT.md sections 18, 46, 62, 83. C63: code + tests only.
You own: engine/learning/contradiction_monitor.py, tests/test_learning_properties.py, tests/test_learning_contradiction_monitor.py.
Import engine/learning modules read-only (do not edit them; S17a/S17b are editing other files now).

1. J09 contradiction monitoring (~250-400 meaningful lines): a monitor that runs every period over the knowledge graph
   (knowledge_graph.py CONTRADICTS edges + contradiction.py investigations): tracks open contradictions over time (born, age,
   investigated, resolved-by-context, unresolved), flags ones growing in evidence or left uninvestigated too long, raises research
   questions (research_priority-compatible records), exports dashboard rows for reports.health_dashboard, never averages two
   contradicting items, respects `now` (nothing dated at/after it). Tests: planted contradiction appears, ages, gets resolved by a
   planted context; an uninvestigated one is escalated; empty graph; future-dated edge refused.
2. I03 property tests (~400-600 lines): use `hypothesis` if installed in .venv (check; if not, write seeded randomized property
   loops with numpy - do not pip install). Properties across modules: stable_hash order/format invariance and numpy-type
   invariance; KnowledgeObject JSON round-trip identity; epistemic transitions never reach an illegal state; belief posterior
   moves toward evidence and never overshoots; hierarchy shrinkage never gives more confidence than the parent at tiny n;
   trader_view never lets any generated year/date form through; curator.release never returns an item whose outcome matured at or
   after real_now; firewalls fail closed on missing fields; transfer_ratio guards (zero/negative denominators) never return inf/nan;
   calibration ECE in [0,1]; determinism: same seed -> identical outputs for planted_world, credit (sampled Shapley), same_year.
   Each property must be able to fail (show one planted counterexample per property is caught). Keep the file under 90 s.
Report ruler counts (scripts/contract_lines.py) and test results.
