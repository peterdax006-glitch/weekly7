# W01_autonomous_research_loop  [model: opus]
C66 sections 3 (autonomous loop, 2,000-3,000), 35 (two-stage architecture), 47 (the full cycle), 50 (autonomous operation),
51 (the controller that knows when to change priorities); canon C64, C67. C63: code + unit/integration tests only.
Read state/build/CONTEXT.md fully (rules 1-27), RESEARCH_BRAIN_CONTRACT.md sections 0-3, 29-31, 34, 35, 47, 50, 51, 53 in full,
state/build/RESEARCH_MAPPING.md, state/build/INTEGRATION.md "C66 wave 1" hooks, and state/build/INTEGRATION_AUDIT.md (the lesson:
text presence is not integration).
You own: engine/research/loop.py, engine/research/two_stage.py, engine/research/controller.py, scripts/research_loop.py,
tests/test_research_loop.py. You may make small API additions (never behaviour changes) to engine/research/* modules if a public
entry is missing; list each.

Build the trusted-side research brain that runs indefinitely:
1. loop.py - the section-3 cycle OBSERVE -> UPDATE KNOWLEDGE -> EVALUATE -> SURPRISES -> FAILURES -> MISSED WINNERS/LOSERS ->
   PATTERN BREAKS -> QUESTIONS -> HYPOTHESES -> INFORMATION GAIN -> ALLOCATE COMPUTE -> RUN EXPERIMENT -> VALIDATE -> CONTROLS ->
   UPDATE KNOWLEDGE -> UPDATE PRIORITIES -> REPEAT, each stage calling the REAL module's public step() (observer, autopsy, missed,
   winners_losers, loss_pipeline, knowability, unknown_cause, counterfactual, break_research, volatility_lab, direction_lab,
   frontier, symmetry, discovery, interactions, multiscale, cross_section, regimes, episodes/precursors (R21), questions, targets,
   hypothesis_tree, priority, experiments, failed_lab, meta_research, compute_manager, value_accounting, waste, replication,
   quality_gate, scorecard, research_graph, decision_bridge, science_memory, brain_health, diversity, firewall). Modules still
   being written (R08 direction_lab, R09 frontier/symmetry, R10 discovery, R15 targets/questions, R16 multiscale/cross_section/
   regimes, R21 episodes) are imported lazily; if absent the stage records SKIPPED_MISSING_MODULE (never silently passes) - and a
   test asserts that once present they are called. Checkpoints, resumability, crash recovery, experiment isolation (via
   compute_manager / engine.learning.compute workers), cancellation/escalation/demotion, duplicate detection, stale-hypothesis
   detection, lineage and reports. A long experiment never blocks the cycle (concurrent jobs where safe; RAM >= 2.5 GB rule).
2. two_stage.py - the section-35 architecture: eligible universe -> point-in-time state -> P(volatility) -> volatility rank ->
   predicted-mover universe -> P(up|mover), P(down|mover) -> calibration -> failure/regime/pattern-health -> risk filter ->
   position/abstention; direction evaluated ONLY on predicted movers; knowledge reaches it only through the research/trader
   firewall (engine.research.firewall) and the curator (C64).
3. controller.py - section 51: continuously compare current vs desired capability, largest uncertainty, largest loss source,
   largest prediction gap, highest-value opportunity, and move research share between the section-51 phases (volatility first;
   predictable-vs-unknowable; direction; 80% at coverage; transfer; catastrophic losses; continuous discovery; improving the
   discovery process), revisiting earlier phases on regression; its allocation feeds diversity/priority/compute_manager.
   Must produce the section-50 statements as data ("volatility is the highest-value unresolved problem, allocate 43%").
4. C67 always-on sweeps: schedule R21's episode sweeps, the volatility lab's sweep and discovery's sweep as permanent low-priority
   background jobs that yield to higher-value work and resume where they stopped.
5. scripts/research_loop.py - the runnable entry (config, root dir, budget, --once / --forever, checkpoint dir), and
   scripts/reachability.py must reach every engine/research module from it (extend the reachability entry points - report the
   checker output; 0 UNREACHED among present modules).
Tests: a full cycle on a planted world runs end to end and produces questions -> priorities -> experiments -> results -> memory
-> new questions (section 47) without future information (planted future leak refused at the right stage); controller shifts
share toward a planted high-value problem and back on regression; a killed loop resumes without redoing or skipping work;
missing-module stages are reported, not passed.
