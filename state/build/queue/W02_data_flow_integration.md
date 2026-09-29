# W02_data_flow_integration  [model: opus]
C66 sections 3, 35, 42, 47 (the full cycle must actually carry data), 53; memory lesson "wired means reachable AND data flows".
Read state/build/CONTEXT.md (rules 1-27), RESEARCH_BRAIN_CONTRACT.md sections 3, 35, 42, 47, 53, state/build/INTEGRATION.md (C66 hooks),
and W01's report facts below. C63: code + tests only (the real-cache feed adapters are code; do not run them on real data).
You OWN: NEW engine/research/feeds.py, NEW engine/research/evidence.py, engine/research/loop.py (feed + evidence plumbing only),
engine/research/science_memory.py (the fold fix only), NEW tests/test_research_dataflow.py. Other modules: small API additions only, listed.

W01 facts (29 Sep): static reachability is 0 unreached, BUT on the planted feed 24 stages report SKIPPED_NO_INPUT (observer, autopsy,
knowability, counterfactual, break_research, discovery, interactions, targets, cross_section, multiscale, precursors, frontier,
symmetry, ...); the quality gate never promotes (it returned NEEDS_MORE_EVIDENCE with 10 gates blocked because the loop only supplies
point-in-time, leak, replication and provenance evidence); science_memory's graph fold is not idempotent when a pattern node's
known_at changes, so the loop skips folding.

Do:
1. feeds.py: ONE feed builder that turns bars (daily OHLCV panels, the shape engine/research/episodes.load_bars returns) plus optional
   events/insider/macro/sector tables into EVERY stage's inputs per simulated day (observer day snapshot, autopsy context, knowability
   frames, counterfactual store, break_research ItemSeries, discovery Panel via inputs_from_wide, interaction inputs, target day input,
   cross-section day frame, multiscale inputs, precursor loader, frontier/symmetry prediction frames from the two-stage outputs...),
   streaming year by year (rule 27), point-in-time (nothing dated at/after now). Two sources: a rich planted world (synthetic_bars +
   planted effects covering every stage) and a real-cache adapter (code only, untested on real files per C63).
2. evidence.py: assemble the FULL evidence bundle the quality gate needs (statistical validity, incremental value, OOS confirmation,
   cross-context transfer, risk, anti-memorisation/identity, future-information audit, reproducibility, stability, provenance,
   calibration, complexity...) from the modules that produce it (replication, transfer, identity_firewall/IdentityHarness,
   firewall audits, symmetry.gate_inputs, calibration, complexity, same_year controls where relevant). No gate is ever passed by
   default - missing evidence stays blocking.
3. science_memory fold: make it idempotent under changing known_at (versioned node history, not overwrite), with a test that folding
   twice == folding once, then turn the fold back on in the loop.
4. The section-47 proof on the rich planted world: a data-flow test that runs the loop N cycles and asserts (a) ZERO stages
   SKIPPED_NO_INPUT, (b) a planted genuine pattern goes question -> experiment -> replication -> quality gate PROMOTE -> firewall ->
   curator/live store -> two_stage decision change -> outcome measured -> new question, (c) a planted NOISE pattern is never promoted,
   (d) a planted future leak is refused at the right stage, (e) the kill-and-resume invariant still holds. Keep test files < 90 s
   (split; mark the long end-to-end run so it can run separately if needed, but it must exist and pass).
Report ruler counts, per-stage input/skip table from the planted run, and test results.
