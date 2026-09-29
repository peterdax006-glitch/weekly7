# P06_c68_into_loop  [model: opus]
Canon C68 checklists T (self-research loop), Y (full error-to-improvement pipeline), Z (the 20 adversarial integration tests) and the
COMPLETION REQUIREMENT; canon C69 sections 4, 12-18, 31 (bare-minimum firewall) and state/build/EXECUTION_LEDGER.md work item W-02.
Read state/build/CONTEXT.md (rules 1-28), PREDICTION_ERROR_ADDITION.md in full, EXECUTION_LEDGER.md sections 0, 2, 4, 6, and
state/build/INTEGRATION.md "C68 (P-series) hooks". C63: code + tests only (planted worlds; no real data).
You OWN: NEW engine/research/error_loop.py, NEW tests/test_research_error_loop.py, NEW tests/test_c68_adversarial.py, and the C68
modules themselves for integration fixes and de-duplication only (expectations, outcomes, prediction_error, error_research,
market_expectations, change_points, regime_memory, pattern_change, what_changed, selection_constraint, exit_research,
calibration_target, self_correct), plus engine/research/two_stage.py (selection + exit wiring). W02 owns engine/research/loop.py,
feeds.py and evidence.py and is adding `loop.register_stage(name, fn, after=...)` for you - register your stages through it and never
edit loop.py; if the hook is not there yet, build against the documented signature and tell the main session.

Build:
1. error_loop.py: the checklist-Y pipeline as registered loop stages - two_stage decision -> P05 selection_constraint (5-10% on
   realisable gain under the intended exit) -> PathModel -> P01 ExpectationLedger.record BEFORE the fill -> P05 learned exit decides
   the exit (never the ±1pp target) -> after maturity P01 outcomes.reconstruct + ErrorEngine -> P02 error_research (depth into the
   EXISTING priority/compute) -> P04 what_changed + pattern_change -> P03 market_expectations/change_points/regime_memory -> hypothesis
   -> replication/quality gate -> promotion or rejection -> monitoring; plus the checklist-T self-research questions each cycle.
   Persistent (state root) and auditable (each record links to its predecessor).
2. De-duplication found by the C69 audit: P03's own small hash chain -> archive ChainFile lanes; two ±1pp statistics (P01
   honest_tolerance vs P05 calibration_target) -> one canonical, the other an adapter; what_changed's five-way split vs knowability ->
   one mapping owned by knowability; pattern_change vs break_research -> pattern_change consumes break_research, no parallel
   detector. State which is canonical and prove the other path is not used.
3. tests/test_c68_adversarial.py: every checklist-Z bullet as a test through the REAL loop on a planted world (predictions cannot be
   rewritten; exits cannot be manipulated; ±1pp never controls selling; no future information in error analysis or regime detection;
   unknowable stays unknowable; tiny errors stay cheap; major errors go deeper; repeated errors escalate; pattern changes detected;
   false regime changes do not disable good patterns; degraded patterns lose influence and recovered ones regain it; market vs single
   stock; exits learned independently; 5-10% enforced; out-of-range stocks cannot be selected to improve another metric; ±1pp cannot
   be gamed; discoveries need OOS; the learner improves only when evidence justifies it).
4. Reachability: scripts/reachability.py --package engine.research must show 0 UNREACHED including all 13 C68 modules.
Report ruler counts, reachability output, the per-stage run table from a planted run, and test results.
