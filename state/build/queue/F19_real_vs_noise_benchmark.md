# F19_real_vs_noise_benchmark  [model: opus]
Canon C70 + C71 (owner, 30 Sep, VERBATIM in canon/CANON.md - read them first), C54-C58, C63, C64, C66 sections 30-31, C69
sections 12-14, 27, 28, 31. Read state/build/CONTEXT.md (rules 1-29).
Read: engine/research/feeds.py (planted_world, PlantConfig, WorldFeed), engine/research/evidence.py (SequentialPlan, gate, train_cut,
failure_floor), engine/research/quality_gate.py, engine/research/discovery.py, discovery_sources.py, volatility_lab.py
(planted_frame), tests/test_regate_sequential.py (sequential_trace / run_loop), state/research/regate_sequential/SUMMARY_clean_loops.md.

GOAL (owner's words): hundreds of simulations; hard-to-notice real patterns and convincing noise; the system must (1) identify ALL
potential patterns, (2) sort real from noise using historic (past-only) evidence, (3) be scored against the truth, and (4) be
iterated until every real pattern is found with zero false positives - in EVERY simulated year/era (C71), while staying blind to the
year (C64).

You OWN: NEW engine/research/pattern_benchmark.py, NEW scripts/pattern_benchmark.py, NEW tests/test_pattern_benchmark.py,
state/research/pattern_benchmark/. Changing discovery/evidence/gate code is a LATER brief: in this brief you build and run the
benchmark and report exactly where the current system fails. Never run git. A real-data Test-loop run (livesim_loop2) is running:
use at most 2 processes of your own and never kill others.

Build:
1. A world generator (reuse feeds.planted_world / PlantConfig where possible; extend via your own module, not by editing feeds.py)
   producing hundreds of worlds from seeds, each with KNOWN truth:
   - REAL patterns at graded strengths, including HARD ones (small effect, sparse, conditional on a context, interaction of two
     features, slowly drifting, regime-limited-but-persistent);
   - CONVINCING NOISE: coincidences that hold strongly in-sample then vanish; multiple-testing winners (many null features, the best
     look great); a pattern that only exists early (decays to zero); a proxy of a real pattern that adds nothing; a period-specific
     fluke; a leak-shaped artefact (correlated with the future only through overlap) that a leak-safe method must reject;
   - ERAS (C71): calm / volatile / crisis / trending / choppy regimes, different base rates and cross-sectional spreads, regime
     shifts inside a world; every era appears many times; the world never reveals its "year".
2. Truth per world: every planted feature labelled REAL / NOISE-kind, its strength and its theoretical detectability (the power an
   oracle test with the true effect would have on that sample). A real pattern below a stated detectability floor is reported as
   UNDETECTABLE-IN-PRINCIPLE, never silently dropped; the floor is computed, not chosen to flatter results.
3. Run the CURRENT system on each world as it would really run (candidate generation -> sequential gate at default settings, past
   data only; the full research loop on a subset if the gate-only path is not representative - say which you used and why).
4. Score, per world and per era, with confidence intervals:
   - CANDIDATE RECALL: share of planted patterns (real AND noise) the system surfaced as candidates at all;
   - FALSE POSITIVES: noise promoted (by noise kind);
   - TRUE-POSITIVE RECALL: real promoted, by strength band (detectable only; undetectable reported separately);
   - time-to-discovery; per-era breakdown; worst era.
5. Split seeds into DEVELOPMENT and HELD-OUT (the held-out set is fixed now, never used for tuning; its seed list is recorded).
   Report both. Run >= 300 worlds in total (state the count and wall time; parallelise within the 2-process limit).
6. Tests (under 90 s): truth labels are exact; a perfect oracle scores 100%/0; a random classifier scores as expected; the
   detectability floor is monotone in effect size; held-out seeds cannot be selected by the tuning code.
7. Report: the score table (development and held-out, per era, per noise kind, per strength band) and a ranked list of the
   concrete failures (which noise kinds get through, which real kinds are missed, in which eras, at which gate/stage) - that list
   drives the next briefs. Nothing is VALIDATED by you.
