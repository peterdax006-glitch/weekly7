# F23_planted_world_trades_and_decision_path  [model: opus]
C75 Phase 1E (complete integration, the chain ends in decision -> outcome -> prediction error -> new question; zero skipped stages),
C68 (expectations/outcomes/errors/what_changed/exits need positions), C66 RF28 (knowledge-to-decision bridge). Read CONTEXT.md
(rules 1-29), state/build/C75_PHASE0_MAPPING.md (gaps 12 and 13), state/research/regate_sequential/SUMMARY_clean_loops.md, F18's
report in the Masterstock journal, engine/research/feeds.py (planted_world), engine/research/two_stage.py, error_loop.py,
decision_bridge.py, engine/adaptive.py (Session), engine/learning/test_path.py.
Findings: (1) the default planted world opens 0 positions in 129/129 cycles in all 9 clean runs - every day's funnel dies at
NOT_PREDICTED_MOVER / LOW_CONFIDENCE / SHORT_DISABLED / OUT_OF_BAND; the world has no mechanism that makes a 5-10% realisable gain
knowable in advance, so C68's expectation->outcome->error chain, what_changed, research_depth and learned exits never run.
(2) knowledge filed by the research loop never reaches the production decision-maker (adaptive.Session).
You OWN: engine/research/feeds.py (planted world: PlantConfig/planted_world only), two_stage.py, decision_bridge.py, and the
decision-path hook in engine/adaptive.py (read CONTEXT rule 29: adaptive.py is on the trader import closure - no research imports
there; the bridge must pass RELEASED knowledge through the existing firewall/curator doors). Never edit evidence.py/quality_gate.py/
loop.py (report defects), engine/research/pattern_benchmark.py (F19), error_loop.py (only if the fix needs it - say so). Never run git.
One process only.
Do:
1. Add to the planted world an honest, knowable-in-advance mechanism that produces 5-10% weekly moves for a minority of names (e.g.
   a persistent volatility state + a directional precursor with a planted, modest edge), keeping the null world free of it, so the
   default two-stage funnel opens positions for the right reason. Never lower a gate threshold to make it trade.
2. Prove on the real loop (<= 40 cycles, default settings): positions open; C68 expectations -> outcomes -> errors -> what_changed ->
   research questions all run (0 SKIPPED_NO_INPUT for those stages after warm-up); the null world still opens ~none and files nothing.
3. Knowledge-to-decision: released knowledge changes a decision of the production decision-maker (adaptive.Session) through the
   firewall/curator path; a test proves the decision changes when the knowledge is present and not otherwise; trader import closure
   stays clean (scripts/leak_audit.py --parts static,assemble: 8d not LEAK).
Report ruler counts, tests, the stage table, and leak-audit result.
